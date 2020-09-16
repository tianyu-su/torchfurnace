# -*- coding: utf-8 -*-
# Date: 2020/3/17 12:14

"""
a tool module for managing model checkpoints, tensorboard, and experiment records
The key idea is to deal with the structure of directory, we only consider what files are needed to save, and then
the file will be saved in the appropriate location.
"""
import subprocess

from torchfurnace.utils.function import log

__author__ = 'tianyu'

import sys
import time
import shutil
from pathlib import Path
from contextlib import contextmanager

from torch.utils.tensorboard import SummaryWriter

from .utils.tracer_component import Config, Model


class Committer(object):
    SNAPSHOT_GIT_NAME = ".snapshootgit"
    SNAPSHOT_GITIGNORE_NAME = ".snapignore"

    def __init__(self, code_dir: Path, work_name: str):
        self._code_dir = code_dir
        self._work_name = code_dir / work_name
        self._snapshoot_git_path = self._code_dir / self.SNAPSHOT_GIT_NAME

    def commit(self):
        self._check_snapshootgit()
        """automatically commit code"""
        try:
            with self._git_switcher(self._code_dir):
                ret = subprocess.call(
                    " && ".join([f"cd {self._code_dir}",
                                 "git add -A",
                                 "git commit -a -m auto_commit"]), shell=True)
                return {"ret": ret, 'commit_id': self._get_commits(self._code_dir / '.git')[-1]}
        except BaseException as e:
            log(msg='Some error occurs during committing.', color='red')
            raise e

    def revert(self, commit_id):
        """copy gaven version to work experiment
            :param commit_id: 回退版本的 commit-id
        """
        flag = False
        for id in self._get_commits(self._snapshoot_git_path):
            if id.startswith(commit_id):
                flag = True
                commit_id = id
                break
        if not flag:
            raise RuntimeError("snaphosted repo don't exist!")

        revert_path = self._work_name / 'revert' / commit_id[:7]
        revert_path.mkdir(parents=True)

        shutil.copytree(self._snapshoot_git_path, (revert_path / self.SNAPSHOT_GIT_NAME))
        try:
            with self._git_switcher(revert_path):
                ret = subprocess.call(
                    " && ".join([f"cd {revert_path}", f"git reset --hard {commit_id}"]), shell=True
                )
            return ret
        except BaseException as e:
            log(msg='Some error occurs during reverting copy process!.', color='red')
            raise e

    def _check_snapshootgit(self):
        if not self._snapshoot_git_path.exists():
            # create git repo for code capturing
            user_gitignore = ''
            if (self._code_dir / '.gitignore').exists():
                user_gitignore = (self._code_dir / '.gitignore').open().read()

            snap_ignore = self._code_dir / self.SNAPSHOT_GITIGNORE_NAME
            snap_ignore.write_text("""
                {user_gitignore}
                # exclude user's git repo
                .git_backup/
                .gitignore_backup
                
                .gitignore
                
                # experiment dirs
                {work_name}/
                
                # some data files
                *.tar*
                *.mdb*
                *.lmdb*
                """.format(user_gitignore=user_gitignore, work_name=self._work_name), encoding='utf-8')

            with self._git_switcher(self._code_dir):
                # init snapshot repo
                ret = subprocess.call("git init", shell=True)
                if ret != 0:
                    raise RuntimeError('git tool maybe not be installed in your system')

    @staticmethod
    def _get_commits(path: Path):
        """从项目目录下的记录获取 snapgit 的所有 commit-id
        """
        lines = (path / 'logs' / 'refs' / 'heads' / 'master').open('r').readlines()
        commit_ids = []
        for line in lines:
            commit_ids.append(line.split()[1])

        return commit_ids

    @contextmanager
    def _git_switcher(self, path: Path):
        self._switch_to_snap_git(path)
        yield
        self._switch_to_standard_git(path)

    def _switch_to_snap_git(self, path: Path):
        """将工作目录从通常的 git 模式切换成 snapgit 模式
        """
        checkout = [('.git', '.git_backup'), ('.gitignore', '.gitignore_backup'),
                    (self.SNAPSHOT_GIT_NAME, '.git'), (self.SNAPSHOT_GITIGNORE_NAME, '.gitignore')]
        for de in checkout:
            if (path / de[0]).exists():
                (path / de[0]).rename((path / de[1]))

    def _switch_to_standard_git(self, path: Path):
        """将工作目录从 snapgit 模式切换成通常的 git 模式
        """
        checkout = [('.git', '.git_backup'), ('.gitignore', '.gitignore_backup'),
                    (self.SNAPSHOT_GIT_NAME, '.git'), (self.SNAPSHOT_GITIGNORE_NAME, '.gitignore')]
        for de in checkout[::-1]:
            if (path / de[1]).exists():
                (path / de[1]).rename((path / de[0]))


class Tracer(object):
    CONFIG_NAME = 'run_config'
    ARCH_NAME = 'architecture'

    def __init__(self, root_dir=Path('.'), work_name='network', clean_up=5):
        self._committer = Committer(code_dir=root_dir, work_name=work_name)
        self._snap_git_switch = True
        self._tb_switch = True
        self._debug_switch = False
        self._dirs = {
            'work_name': root_dir / work_name
        }
        self._clean_up_top = clean_up

    def _start_log(self, logger_name):
        # redirect stdout stderr to file
        self._log = open(logger_name, 'w', encoding='utf-8')
        self._stderr = sys.stderr
        self._stdout = sys.stdout
        sys.stderr = self._log
        sys.stdout = self._log

    def _clean_up(self):
        # remain Top5 best model checkpoint
        files = self._dirs['checkpoint_best'].glob('{}*'.format(self._clean_up_prefix))
        import re, os
        files = sorted(files, key=lambda x: float(re.findall(r'Acc(.*?)_', str(x))[0]), reverse=True)
        for file in files[self._clean_up_top:]:
            os.remove(file)

    def dirs(self, path_key):
        assert self._dirs.__contains__(path_key), f"{path_key} is wrong path_name."
        return self._dirs.get(path_key)

    def close(self):
        if self._clean_up_top > 0:
            self._clean_up()
        # close I/O
        if self._tb_switch: self._tb.close()
        if not self._debug_switch:
            self._log.close()
            # recovery stdout stderr to system
            sys.stderr = self._stderr
            sys.stdout = self._stdout

    @property
    def tb(self):
        # expose to caller
        return self._tb

    @staticmethod
    def _get_now_time():
        return time.strftime('%m%d_%H-%M-%S', time.localtime(time.time()))

    def _build_dir(self):
        for k, d in self._dirs.items():
            if k == 'experiment_name': continue
            if not d.exists(): d.mkdir(parents=True)

    def snap_git_switch(self, status):
        self._snap_git_switch = status
        return self

    def tb_switch(self, status):
        self._tb_switch = status
        return self

    def debug_switch(self, status):
        self._debug_switch = status
        return self

    def attach(self, experiment_name='exp', logger_name='log', override=True):
        if not override:
            experiment_name += f"_{Tracer._get_now_time()}"
        self._dirs['experiment_name'] = experiment_name
        if self._tb_switch:
            self._dirs['tensorboard'] = self._dirs['work_name'] / 'tensorboard' / f"{self._dirs['experiment_name']}"
        self._dirs['models'] = self._dirs['work_name'] / 'models' / self._dirs['experiment_name']
        self._dirs['checkpoint_best'] = self._dirs['models'] / 'checkpoint' / 'best'
        self._dirs['logs'] = self._dirs['work_name'] / 'logs' / self._dirs['experiment_name']

        self._build_dir()

        # edit tensorboard log_dir
        if self._tb_switch:
            self._tb = SummaryWriter(log_dir=self._dirs['tensorboard'])
            print(f"Start Tensorboard ... [tensorboard --port=6006 --logdir {self._dirs['tensorboard']}]")

        # new log file
        logger_name = self._dirs['logs'] / logger_name
        if logger_name.exists(): logger_name = f'{logger_name}_{self._get_now_time()}'

        if not self._debug_switch: self._start_log(logger_name)

        # automatically generate a readme for recording something
        (self._dirs['models'] / 'readme.txt').open('w+', encoding='utf-8')

        # expose dirs
        return self

    def store(self, component):
        if isinstance(component, Config):
            if self._snap_git_switch:
                # add snap commit id
                log(msg="==== auto commit ====", color='blue')
                try:
                    state = self._committer.commit()
                    if state['ret'] == 0:
                        component.add_item('snap_commit_id', state['commit_id'])
                        log(msg=f"commit id: {state['commit_id']}")
                    else:
                        log(msg=f"some errors in the proecess of commiting!", color='red')
                except Exception as e:
                    log(msg='auto commit fail!', color='red')
                    log(msg='+++ Trace +++')
                    log(msg=str(e))

            component.save(self._dirs['models'] / f'{Tracer.CONFIG_NAME}.json')
        if isinstance(component, Model):
            self._clean_up_prefix = component.name.replace('.pth.tar', '')
            component.save(self._dirs['checkpoint_best'].parent, self._dirs['models'] / f'{Tracer.ARCH_NAME}.txt')

    def load(self, component):
        if isinstance(component, Model):
            return component.load(self._dirs['work_name'] / 'models')

    def revert(self, commit_id):
        self._committer.revert(commit_id)

# global var
# 是否显示日志信息
import platform

SHOW_LOG = True


def get_platform():
    '''获取操作系统名称及版本号'''
    return platform.platform()


def get_version():
    '''获取操作系统版本号'''
    return platform.version()


def get_architecture():
    '''获取操作系统的位数'''
    return platform.architecture()


def get_machine():
    '''计算机类型'''
    return platform.machine()


def get_node():
    '''计算机的网络名称'''
    return platform.node()


def get_processor():
    '''计算机处理器信息'''
    return platform.processor()


def get_system():
    '''获取操作系统类型'''
    return platform.system()


def get_uname():
    '''汇总信息'''
    return platform.uname()


def get_python_build():
    ''' the Python build number and date as strings'''
    return platform.python_build()


def get_python_compiler():
    '''Returns a string identifying the compiler used for compiling Python'''
    return platform.python_compiler()


def get_python_branch():
    '''Returns a string identifying the Python implementation SCM branch'''
    return platform.python_branch()


def get_python_implementation():
    '''Returns a string identifying the Python implementation. Possible return values are: ‘CPython’, ‘IronPython’, ‘Jython’, ‘PyPy’.'''
    return platform.python_implementation()


def get_python_version():
    '''Returns the Python version as string 'major.minor.patchlevel'
    '''
    return platform.python_version()


def get_python_revision():
    '''Returns a string identifying the Python implementation SCM revision.'''
    return platform.python_revision()


def get_python_version_tuple():
    '''Returns the Python version as tuple (major, minor, patchlevel) of strings'''
    return platform.python_version_tuple()


def show_os_all_info():
    '''打印os的全部信息'''
    print('获取操作系统名称及版本号 : [{}]'.format(get_platform()))
    print('获取操作系统版本号 : [{}]'.format(get_version()))
    print('获取操作系统的位数 : [{}]'.format(get_architecture()))
    print('计算机类型 : [{}]'.format(get_machine()))
    print('计算机的网络名称 : [{}]'.format(get_node()))
    print('计算机处理器信息 : [{}]'.format(get_processor()))
    print('获取操作系统类型 : [{}]'.format(get_system()))
    print('汇总信息 : [{}]'.format(get_uname()))


def show_os_info():
    '''只打印os的信息，没有解释部分'''
    print(get_platform())
    print(get_version())
    print(get_architecture())
    print(get_machine())
    print(get_node())
    print(get_processor())
    print(get_system())
    print(get_uname())


def show_python_all_info():
    '''打印python的全部信息'''
    print('The Python build number and date as strings : [{}]'.format(get_python_build()))
    print('Returns a string identifying the compiler used for compiling Python : [{}]'.format(get_python_compiler()))
    print('Returns a string identifying the Python implementation SCM branch : [{}]'.format(get_python_branch()))
    print('Returns a string identifying the Python implementation : [{}]'.format(get_python_implementation()))
    print('The version of Python ： [{}]'.format(get_python_version()))
    print('Python implementation SCM revision : [{}]'.format(get_python_revision()))
    print('Python version as tuple : [{}]'.format(get_python_version_tuple()))


def show_python_info():
    '''只打印python的信息，没有解释部分'''
    print(get_python_build())
    print(get_python_compiler())
    print(get_python_branch())
    print(get_python_implementation())
    print(get_python_version())
    print(get_python_revision())
    print(get_python_version_tuple())


def test():
    print('操作系统信息:')
    if SHOW_LOG:
        show_os_all_info()
    else:
        show_os_info()
    print('#' * 50)
    print('计算机中的python信息：')
    if SHOW_LOG:
        show_python_all_info()
    else:
        show_python_info()


def init():
    global SHOW_LOG
    SHOW_LOG = True


def main():
    init()
    test()


if __name__ == '__main__':
    # main()

    import hashlib
    m = hashlib.md5()
    m.update((platform.platform() + platform.node()).encode("utf8"))
    print(m.hexdigest())

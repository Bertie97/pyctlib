#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################

__all__ = """
    executor
    shell
""".split()

import re, os, sys
from pyoverload import overload, params

class executor:

    @params
    def __init__(self, home:str='', verbose:bool=False):
        self.verbose = verbose
        self.working_dir = os.path.abspath(os.path.curdir)
        if not home: return
        self.working_dir = home
        self.toolsList = []
        for file in os.listdir(home):
            if os.path.isdir(file): continue
            self.toolsList.append(file.split('.')[0])
            self.__dict__[file.split('.')[0]] = os.path.join(home, file)

    @params
    def __getattr__(self, string:str):
        def run(x, variables={}, force=False):
            self(self.__dict__.get(string, string) + x, variables, force)
        return run

    @overload
    def __call__(self, string:str, force:bool = False, **variables):
        local_variables = sys._getframe().f_back.f_locals
        local_variables.update(variables)
        cmd = string.format(**local_variables)
        if self.verbose: print("Running:", cmd)
        tmpcmd = string.format(**{k: v.replace(' ', '\ ') if isinstance(v, str) else v for k, v in local_variables.items()})
        do, *args = re.sub(r'[^\\] ', lambda x: x.group().replace(' ', '\n'), tmpcmd).split('\n')
        if self.__dict__.get(do, do) != do: return self(cmd.replace(do, self.__dict__.get(do, do)))
        for i, arg in enumerate(args):
            arg = arg.strip()
            if os.path.sep in arg:
                opt = args[i-1].strip()
                arg = arg.replace('\ ', ' ')
                arg = arg.strip('\'"')
                if not os.path.exists(arg):
                    if opt != '-o' and os.path.sep in ''.join(args[i + 1:]):
                        print('\n', cmd, sep='')
                        if not force: raise AssertionError("Path doesn't exists: " + arg)
                        else: print("Warning: Path doesn't exists: " + arg)
                    elif not os.path.exists(os.path.dirname(arg)): path(arg) * Folder
                elif opt == '-o' and not force: return
        if self.verbose: os.system(cmd)
        else: return os.popen(cmd).read()

    @overload
    def __call__(self, string:str, variables:dict={}, force:bool=False):
        return self(string, force, **variables)

    def set_wd(self, working_dir:str=''):
        if working_dir: self.working_dir = working_dir
        else: self.working_dir = os.path.abspath(os.path.curdir)

shell = executor()

if __name__ == '__main__': 
    shell = executor()

    shell.set_wd()
    print(shell("ls"))

class SPrint:
    """
    Print to a string.

    example:
    ----------
    >>> output = SPrint("!>> ")
    >>> output("Use it", "like", 'the function', "'print'.", sep=' ')
    !>> Use it like the function 'print'.
    >>> output("A return is added automatically each time", end=".")
    !>> Use it like the function 'print'.
    A return is added automatically each time.
    >>> output.text
    !>> Use it like the function 'print'.
    A return is added automatically each time.
    """

    def __init__(self, init_text=''):
        self.text = init_text

    def __call__(self, *parts, sep=' ', end='\n'):
        if not parts: end = ''
        self.text += sep.join([str(x) for x in parts if str(x)]) + end
        return self.text

    def __str__(self): return self.text

# class TOOLS:
#     def __init__(self, homeDIR='', verbose=False):
#         self.delete = "rm -r" if 'darwin' in sys.platform.lower() or 'linux' in sys.platform.lower() else "del"
#         self.verbose = verbose
#         if not homeDIR: return
#         self.toolsList = []
#         for file in os.listdir(homeDIR):
#             if os.path.isdir(file): continue
#             self.toolsList.append(file.split('.')[0])
#             self.__dict__[file.split('.')[0]] = os.path.join(homeDIR, file)

#     def __getattr__(self, string):
#         def run(x, variables={}, force=False):
#             self(self.__dict__.get(string, string) + x, variables, force)
#         return run

#     def __call__(self, string, *args, **variables):
#         force = False
#         if len(args) >= 2: raise SyntaxError("Too much parameters for shell exchange! ")
#         if 'force' in variables: force = variables['force']; variables.pop('force')
#         if len(args) > 0: assert len(variables) == 0; variables = args[0]
#         if 'variables' in variables: variables = variables['variables']
#         cmd = string.format(**variables)
#         if self.verbose: print("Running:", cmd)
#         tmpcmd = string.format(**{k: v.replace(' ', '\ ') if isinstance(v, str) else v for k, v in variables.items()})
#         do, *args = re.sub(r'[^\\] ', lambda x: x.group().replace(' ', '\n'), tmpcmd).split('\n')
#         if self.__dict__.get(do, do) != do: return self(cmd.replace(do, self.__dict__.get(do, do)))
#         for i, arg in enumerate(args):
#             arg = arg.strip()
#             if os.path.sep in arg:
#                 opt = args[i-1].strip()
#                 arg = arg.replace('\ ', ' ')
#                 arg = arg.strip('\'"')
#                 if not os.path.exists(arg):
#                     if opt != '-o' and os.path.sep in ''.join(args[i + 1:]):
#                         print('\n', cmd, sep='')
#                         if not force: raise AssertionError("Path doesn't exists: " + arg)
#                         else: print("Warning: Path doesn't exists: " + arg)
#                     elif not os.path.exists(os.path.dirname(arg)): path(arg) * Folder
#                 elif opt == '-o' and not force: return
#         if self.verbose: os.system(cmd)
#         else: return os.popen(cmd).read()

# shell = TOOLS()

# class tmpFile:
#     def __init__(self, *path, preserve=False): self.file_list = list(path); self.update(path); self._p = preserve
#     def __call__(self, *path): self.file_list.extend(path); self.update(path); return path[0]
#     def __del__(self): [os.remove(f) for f in self.file_list if os.path.exists(f) and not self._p]
#     def update(self, flist): [path(f).mkdir() for f in flist]

# class trans_cmd(str):
#     def __new__(cls, string, *transformations, source = None, target = None, verbose = False):
#         if "zxhtransform" in string: return super().__new__(cls, string)
#         assert source and target
#         transformations = (string,) + transformations
#         cmd = ["zxhtransform", target, "{source}", "-o {outfilename}", "-n", str(len(transformations))]
#         for t in transformations: cmd.extend(['-t', t])
#         if not verbose: cmd.append("-v 0")
#         self = super().__new__(cls, ' '.join(cmd))
#         self.target_space = (source, target)
#         return self
        
#     def __add__(x, y): return trans_cmd(str(x).rstrip(' ') + ' ' + y.lstrip(' '))
#     def __invert__(self):
#         _, target, source, _, outfilename, *_ = self.split()
#         target = self.target_space[(self.target_space.index(target) + 1) % 2]
#         transformations = [x.strip() for x in re.findall(r'-t ([^ ]+)', self)]
#         options = [x.strip() for x in re.findall(r' (-[^-]+)', self) if x[1] not in "not"]
#         options += [x.strip() for x in re.findall(r' (--[^-]+)', self)]
#         newcmd = ["zxhtransform", target, source, "-o", outfilename, "-n", str(len(transformations))]
#         for t in transformations[::-1]: newcmd.extend(['-t', t])
#         if '-forward' in options:
#             while '-forward' in options: options.remove('-forward')
#         else: options.append('-forward')
#         newcmd.extend(options)
#         return trans_cmd(' '.join(newcmd))

#     def __call__(self, fin, fout, **kwargs):
#         shell(self, source = fin, outfilename = fout, **kwargs)

#     def transformations(self, i = None):
#         T = [x.strip() for x in re.findall(r'-t ([^ ]+)', self)]
#         if i is None: return T
#         else: return T[i]
#     def replace(self, *args): return trans_cmd(super().replace(*args))

# def registration(code, target, source, output = '', 
#     save_images = False, save_transforms = True, variables = {}, 
#     verbose = False, trans_only = False, reg_file_format = lambda s: os.path.join('reg_files', s)):
#     '''
#     code with format:
#     rigid -Reg 2 -sub 10 10 10 -sub 8 8 8 -length 3 1.5 -steps 200 200
#     affine -Reg 2 -sub 8 8 8 -sub 6 6 6 -length 1 0.5 -steps 200 50
#     FFD -Reg 3 -ffd 80 80 80 -sub 8 8 8 -ffd 40 40 40 -sub 6 6 6 -ffd 20 20 20 -sub 6 6 6 -length 2 2 1 -steps 200 -bending 0.0001
#     '''
#     shell = TOOLS(verbose = verbose)
#     lines = [l.strip() for l in code.split('\n') if l.strip()]
#     outfile = '', 'NONE'
#     name, ext = 0, 1
#     operations = {'rigid': 'AFF', 'affine': 'AFF', 'ffd': 'FFD'}
#     header = {'rigid': "zxhreg -rig", 'affine': "zxhregaff -aff", 'ffd': "zxhregffdsemi0"}
#     new_source = reg_file_format(path(source) - path(source) ** path(target))
#     sourcefile = new_source
#     tmp_images = tmpFile(preserve = save_images)
#     tmp_transforms = tmpFile(preserve = save_transforms)
#     transformations = []
#     for i, line in enumerate(lines):
#         do = line.split()[0]
#         if outfile[ext] == 'NONE': pass
#         elif outfile[ext] == operations[do.lower()]:
#             transformations.pop(-1)
#         else: sourcefile = outfile[name] + '.nii.gz'
#         outfile = '.'.join((sourcefile, do.lower())), operations[do.lower()]
#         transformations.append('.'.join(outfile))
#     sourcefile = new_source
#     outfile = '', 'NONE'
#     if trans_only:
#         for t in transformations:
#             if not os.path.isfile(t):
#                 trans_only = False
#                 break
#     for i, line in enumerate(lines):
#         do = line.split()[0]
#         s = line.find(do)
#         options = (line[:s] + line[s+len(do):]).strip()
#         if outfile[ext] == 'NONE':
#             if '-pre' not in line: prealign = '-pre 13'
#             else:
#                 prealign = re.findall(r' +-pre +[1-9]+| +-pre +0 +[^ ]+', options)[-1]
#                 options = options.replace(prealign, '')
#                 prealign = prealign.strip()
#         elif outfile[ext] == operations[do.lower()]:
#             prealign = '-pre 0 ' + '.'.join(outfile)
#         else: prealign = ''
#         if not prealign: sourcefile = outfile[name] + '.nii.gz'
#         outfile = '.'.join((sourcefile, do.lower())), operations[do.lower()]
#         tmp_images(outfile[name] + '.nii.gz')
#         tmp_transforms('.'.join(outfile))
#         if sourcefile == new_source: sfile = source
#         else: sfile = sourcefile
#         cmd = ' '.join((header[do.lower()], "-target", target,
#                         "-source %s -o %s"%(sfile, outfile[name]),
#                         options, prealign)).strip()
#         if i >= len(lines) - 1 or operations[lines[i+1].split()[0].lower()] != outfile[ext]: pass
#         else: cmd += " -notsaveimage"
#         if not verbose: cmd += " -v 0"
#         if not trans_only:
#             shell(cmd, variables, force = True)
#             if i >= len(lines) - 1 and output: shell("cp %s %s"%(outfile[name] + '.nii.gz', output))
#     return trans_cmd(*transformations[::-1], source = source, target = target, verbose = verbose)

# if __name__ == '__main__': pass

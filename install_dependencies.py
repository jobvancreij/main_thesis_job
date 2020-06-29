import os
import sys
import platform
github_repos_installs = [
    "jobvancreij/LJT_errors",
    "jobvancreij/LJT_database",
    "jobvancreij/LJT_helper_functions",
    "jobvancreij/LJT_pubsub_codes",
    "jobvancreij/deep_learning_models",
    "jobvancreij/trading_algorithms"

]
#check if correct python version. Don't install otherwise will result in errors later on because python3.6 style used
if sys.version_info[0] < 3:
    raise Exception("You are using a python version < 3, use at least 3.6")
elif sys.version_info[1] < 6:
    raise Exception("Run this file with Python 3.6 or greater")
else:
    print("Python version accepted")

#Check if it is running on windows or other platform. Since this needs other install
if 'windows' in platform.platform().lower():
    windows = True
else:
    windows = False
print(f"Running on {platform.platform()}, so running on windows = {windows}")


if windows: #no sudo install on windows
    install = os.system(f"python -m pip install --upgrade -r requirements.txt")
    if install != 0: #install requirements windows
        raise TypeError("Pip installs did not finish correctly. Check requirements.txt and install method")
    for repo in github_repos_installs: #install github repos windows
        install = os.system(f'python -m pip install --upgrade git+https://git@github.com/{repo}.git')
        if install != 0:
            raise TypeError(f"Github repo installs did not finish correctly. Check install {repo}")
    #install_2 = os.system("python -m pip install --upgrade tensorflow-gpu==2.0.0") #at the end since other version can be installed earlier
    #if install_2 != 0: #install requirements windows
        #raise TypeError("tensorflow not installed correctly")

else: #on the server the pip install have to be done via sudo
    install = os.system(f"sudo python3.6 -m pip install --upgrade -r requirements.txt")
    if install != 0: #install requirements linux and debian
        raise TypeError("pip installs did not finish correctly. Check requirements.txt and install method")
    for repo in github_repos_installs: #install github linux and debian
        install = os.system(f'sudo python3.6 -m pip install --upgrade git+https://github.com/{repo}.git')
        if install != 0:
            raise TypeError(f"Github repo installs did not finish correctly. Check install {repo}")
    #tensorflow 2.1 used on servers with cuda 10.1
    install_2 = os.system("sudo python3.6 -m pip install --upgrade tensorflow-gpu==2.1.0")
    if install_2 != 0: #install requirements linux and debian
        raise TypeError("tensorflow not installed correctly")


print('----------------------------------------------------------------')
print("Succesfully installed all packages")

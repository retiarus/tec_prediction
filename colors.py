class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def print_blue(*args):
    print(bcolors.OKBLUE, *args, bcolors.ENDC)


def print_green(*args):
    print(bcolors.OKGREEN, *args, bcolors.ENDC)


def print_red(*args):
    print(bcolors.FAIL, *args, bcolors.ENDC)

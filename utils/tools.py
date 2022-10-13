import os

def log(string_list, log_path, print_console=True, clear=False):
    if log_path is None:
        if print_console:
            for s in string_list:
                print(s)
        return
    if clear and  os.path.exists(log_path):
        os.remove(log_path)

    with open(log_path, 'a+') as log_writer:
        for s in string_list:
            if print_console:
                print(s)
            log_writer.write(s+os.linesep)

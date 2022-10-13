import os

def log(string_list, log_path, print_console=True, clear=False):
    if log_path == None:
        if print_console:
            for s in string_list:
                print(s)
        return
    if clear:
        os.remove(log_path)
    with open(log_path, 'a+') as log_writer:
        for s in string_list:
            if print_console:
                print(s)
            log_writer.write(s+os.linesep)
        log_writer.flush()
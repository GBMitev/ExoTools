def run_simple_command(cmd, splitlines = True):
    from subprocess import run

    result = run(cmd, shell = True, capture_output=True, text=True)
    if splitlines == True:
        return result.stdout.splitlines()
    else:
        return result
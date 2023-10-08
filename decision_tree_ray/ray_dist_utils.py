def hi():
    import os
    import socket
    return f"Running on {socket.gethostname()} in pid {os.getpid()}"

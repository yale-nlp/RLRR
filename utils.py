import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import socket


class Recorder():
    def __init__(self, id=None, log=True, base_dir="ckpts", name=None):
        assert id is not None or name is not None
        assert not (id is not None and name is not None)
        self.log = log
        if name is not None:
            self.dir = name
        else:
            now = datetime.now()
            date = now.strftime("%y-%m-%d")
            self.dir = os.path.join(base_dir, f"{date}-{id}")
        if self.log:
            if id is not None or not os.path.exists(self.dir):
                os.mkdir(self.dir)
            self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.writer = SummaryWriter(os.path.join(self.dir, "log"), flush_secs=60)
            
    def write_config(self, args, models, name):
        if self.log:
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                print(name, file=f)
                print(args, file=f)
                print(file=f)
                for (i, x) in enumerate(models):
                    print(x, file=f)
                    print(file=f)
        print(args)
        print()
        for x in models:
            print(x)
            print()

    def print(self, x=None):
        if x is not None:
            print(x, flush=True)
        else:
            print(flush=True)
        if self.log:
            if x is not None:
                print(x, file=self.f, flush=True)
            else:
                print(file=self.f, flush=True)

    def plot(self, tag, values, step):
        if self.log:
            self.writer.add_scalars(tag, values, step)


    def __del__(self):
        if self.log:
            self.f.close()
            self.writer.close()

    def save(self, model, name):
        if self.log:
            torch.save(model.state_dict(), os.path.join(self.dir, name))

    def save_pretrained(self, model, name, **kwargs):
        if self.log:
            model.save_pretrained(os.path.join(self.dir, name), **kwargs)


def read_json(file_path: str) -> list[dict]:
    """
    Read a JSON/JSONL file and return its contents as a list of dictionaries.
    
    Parameters:
        file_path (str): The path to the JSON file.
        
    Returns:
        list[dict]: The contents of the JSON file as a list of dictionaries.
    """
    try:
        with open(file_path) as f:
            data = [json.loads(x) for x in f]
        return data
    except json.decoder.JSONDecodeError:
        with open(file_path) as f:
            data = json.load(f)
        return data
    
def is_port_in_use(port):
    """
    Check if a port on localhost is in use.

    Args:
        port (int): The port number to check.

    Returns:
        bool: True if the port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Attempt to bind to the port
            s.bind(('localhost', port))
            return False  # Port is available
        except socket.error:
            return True   # Port is in use
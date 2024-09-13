"""Web server for camera visualization."""
from absl import app
from absl import flags
from absl import logging

import functools
import http.server
import pathlib
import socket
import socketserver

flags.DEFINE_integer('port', 9091, 'port for http server.')
FLAGS = flags.FLAGS


class AddressReusableServer(socketserver.TCPServer):
  allow_reuse_address = True


def main(argv):
  del argv  # unused
  handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                              directory=str(
                                  pathlib.Path(__file__).parent.resolve()))
  with AddressReusableServer(("", FLAGS.port), handler) as httpd:
    logging.info("Server started at http://localhost:{}".format(FLAGS.port))
    httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    httpd.serve_forever()


if __name__ == "__main__":
  app.run(main)

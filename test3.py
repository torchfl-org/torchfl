import json
from json import load, loads

from http.server import BaseHTTPRequestHandler
import urllib.parse

class GetHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        tainted = urlparse.urlparse(self.path).query

        # ruleid: tainted-json
        json.load(tainted)
        # ruleid: tainted-json
        json.load(tainted, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)
        # ruleid: tainted-json
        json.loads(tainted)
        # ruleid: tainted-json
        json.loads(tainted, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)

        decoder = json.JSONDecoder()
        # ruleid: tainted-json
        decoder.decode(tainted)
        # ruleid: tainted-json
        decoder.raw_decode(tainted)

        # ok: tainted-json
        json.load(s)
        # ok: tainted-json
        json.load(s, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)
        # ok: tainted-json
        json.loads(s)
        # ok: tainted-json
        json.loads(s, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)

        decoder = json.JSONDecoder()
        # ok: tainted-json
        decoder.decode(s)
        # ok: tainted-json
        decoder.raw_decode(s)

        # ruleid: tainted-json
        load(tainted)
        # ruleid: tainted-json
        load(tainted, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)
        # ruleid: tainted-json
        loads(tainted)
        # ruleid: tainted-json
        loads(tainted, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)

        decoder = JSONDecoder()
        # ruleid: tainted-json
        decoder.decode(tainted)
        # ruleid: tainted-json
        decoder.raw_decode(tainted)

        # ok: tainted-json
        load(s)
        # ok: tainted-json
        load(s, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)
        # ok: tainted-json
        loads(s)
        # ok: tainted-json
        loads(s, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None)

        decoder = JSONDecoder()
        # ok: tainted-json
        decoder.decode(s)
        # ok: tainted-json
        decoder.raw_decode(s)


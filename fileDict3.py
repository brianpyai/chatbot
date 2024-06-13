import sqlite3
import time
import atexit
import os,mimetypes,re
from urllib.parse import quote, unquote
from fastapi import FastAPI, Response, Request
from  fastapi.responses import StreamingResponse,HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles

import mimetypes
mimetypes.add_type("video/webm" ,'.mkv')
mimetypes.add_type("audio/flac",".flac")
mimetypes.add_type("text/plain","..ass")

class FileDict:
    def __init__(self, file_path=":memory:", buffer_size=1000, buffer_idle_time=5, table='filedict'):
        self.file_path = file_path
        self.table = table
        self.conn = sqlite3.connect(file_path,check_same_thread=False)
        self.conn.execute('CREATE TABLE IF NOT EXISTS {} (key TEXT PRIMARY KEY, value TEXT)'.format(self.table))

        self.buffer = []
        self.buffer_size = buffer_size
        self.last_commit_time = time.time()
        self.buffer_idle_time = buffer_idle_time
        atexit.register(self.close)

    def get(self, key):
        try:return self.__getitem__(key)
        except KeyError: return None
    
    def __getitem__(self, key):
        self._check_buffer()
        cursor = self.conn.execute('SELECT value FROM {} WHERE key = ?'.format(self.table), (key,))
        result = cursor.fetchone()
        if result is None:
            raise KeyError(key)
        return result[0]

    def Tables(self):
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        return table_names
    def __setitem__(self, key, value):
        try:
            self.check_key(key)
            self.buffer.append(('set', key, value))
            self._check_buffer()
        except sqlite3.IntegrityError:
            self.buffer.append(('update', key, value))
            self._check_buffer()

    def __delitem__(self, key):
        self.buffer.append(('del', key))
        self._check_buffer()

    def __iter__(self):
        self._check_buffer()
        cursor = self.conn.execute('SELECT key FROM {}'.format(self.table))
        while True:
            result=cursor.fetchone()
            if not result or result is None: break
            yield result[0]
        cursor.close()
        #raise StopIteration
        

    def items(self):
        self._check_buffer()
        cursor = self.conn.execute('SELECT key, value FROM {}'.format(self.table))
        while True:
            result=cursor.fetchone()
            if not result or result is None: break
            yield result
            
        cursor.close()
        #raise StopIteration
        #return cursor.fetchall()

    def from_dict(self, dict):
        self.check_dict(dict)

        self.conn.execute('DROP TABLE IF EXISTS {}'.format(self.table))
        self.conn.execute('CREATE TABLE {} (key TEXT PRIMARY KEY, value TEXT)'.format(self.table))
        self.conn.executemany('INSERT INTO {} (key, value) VALUES (?, ?)'.format(self.table), dict.items())
        self.conn.commit()

    def add_items(self, items):
        for key, value in items.items():
            try:
                self.check_key(key)
                self.buffer.append(('set', key, value))
                self._check_buffer()
            except sqlite3.IntegrityError:
                self.buffer.append(('update', key, value))
                self._check_buffer()
        self._check_buffer()

    def _check_buffer(self):
        if not self.buffer:
            return
        idle_time = time.time() - self.last_commit_time
        if len(self.buffer) >= self.buffer_size or idle_time >= self.buffer_idle_time:
            self._commit()

    def _commit(self):
        if not self.buffer:
            return 
        cursor = self.conn.cursor()
        for op in self.buffer:
            if op[0] == 'set':
                cursor.execute('INSERT OR REPLACE INTO {} (key, value) VALUES (?, ?)'.format(self.table), (op[1], op[2]))
            elif op[0] == 'update':
                cursor.execute('UPDATE {} SET value = ? WHERE key = ?'.format(self.table), (op[2], op[1]))
            elif op[0] == 'del':
                cursor.execute('DELETE FROM {} WHERE key = ?'.format(self.table), (op[1],))
        self.buffer = []
        self.last_commit_time = time.time()
        self.conn.commit()

    def check_dict(self, dictionary):
        for key in dictionary:
            self.check_key(key)

    def check_key(self, key):
        if not isinstance(key, str):
            raise TypeError('Keys must be strings.')
        if not key:
            raise ValueError('Keys cannot be empty strings.')

    def search_keys(self, pattern, like=True, values=False,limited=-1):
        self._check_buffer()
        operator = 'LIKE' if like else '='
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT key FROM {self.table} WHERE key {operator} ?", (pattern,))
        while True:
            result=cursor.fetchone()
            if not result or result is None: break
            yield result[0]
        cursor.close()    
    def search_values(self, pattern, like=True, values=False,limited=-1):
        self._check_buffer()
        operator = 'LIKE' if like else '='
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT key FROM {self.table} WHERE value {operator} ? LIMIT ?", (pattern,limited ))
        while True:
            result=cursor.fetchone()
            if not result or result is None: break
            yield result[0]
        cursor.close()
    def close(self):       
        self._commit()
        try:
            self.conn.commit()
        except:
            pass
        self.conn.close()


import os,json
import sqlite3
from io import BytesIO,StringIO
import uuid
import tempfile
from datetime import datetime
import mimetypes
mimetypes.add_type("video/webm" ,'.mkv')
mimetypes.add_type("audio/flac",".flac")
   
DEFAULT_BLOCK_SIZE = 1024*1024*8 

class FileS:
    def __init__(self, meta, conn):
        self.size = meta['length']
        self.create = meta['created']
        self.modified = meta['modified']
        self.mimetype = meta['mimetype']
        self.encoding = meta['encoding']
        self.parts = json.loads(meta['parts'])
        self.conn = conn
        self.position = 0
        self.buffer = [b'', -1, -1]

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        data = b''
        while size > 0:
            if size < DEFAULT_BLOCK_SIZE and self.buffer[1] <= self.position < self.buffer[2]:
                chunk = self.buffer[0]
                start = self.position - self.buffer[1]
                end = min(start + size, self.buffer[2] - self.buffer[1])
                data += chunk[start:end]
                size -= end - start
                self.position += end - start
            else:
                
                part = self._get_next_part()
                if not part:
                    break
                cur = self.conn.cursor()
                cur.execute('SELECT data FROM datas WHERE uuid=?', (part['uuid'],))
                chunk = cur.fetchone()[0]
                if size >= DEFAULT_BLOCK_SIZE:
                    start = self.position % DEFAULT_BLOCK_SIZE
                    end = start + DEFAULT_BLOCK_SIZE
                    data += chunk[start:end]
                    size -= end - start
                    self.position += end - start
                else:
                    
                    chunk_start = self.position // DEFAULT_BLOCK_SIZE * DEFAULT_BLOCK_SIZE
                    chunk_end = min(chunk_start + DEFAULT_BLOCK_SIZE, part['end'])
                    chunk_pos_start = chunk_start - part['start']
                    chunk_pos_end = chunk_end - part['start']
                    self.buffer = [chunk[chunk_pos_start:chunk_pos_end], chunk_start, chunk_end]
                    start = self.position - chunk_start
                    end = min(start + size, chunk_end - chunk_start)
                    data += chunk[chunk_pos_start+start:chunk_pos_start+end]
                    size -= end - start
                    self.position += end - start
        return data

    def _get_next_part(self):
        for part in self.parts:
            if self.position < part['end'] and self.position >= part['start']:
                return part
        return None

    def seek(self, position):
        self.position = position
        self.buffer = [b'', -1, -1]

    def tell(self):
        return self.position





class FileSQL3:

    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path,check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS files (
                        path TEXT PRIMARY KEY,
                        created TEXT,
                        modified TEXT,
                        length INTEGER,
                        encoding TEXT,
                        mimetype TEXT,
                        description TEXT,
                        parts TEXT)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS datas (
                        uuid TEXT PRIMARY KEY,
                        data BLOB,
                        path TEXT,
                        start INTEGER,
                        end INTEGER)''')
        self.conn.commit()

    def get(self, file_path):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM files WHERE path=?', (file_path,))
        meta = cur.fetchone()
        if meta:
            return FileS(dict(meta), self.conn)

    def putBytes(self, b,p_path,**kws):
        f=tempfile.NamedTemporaryFile(delete=False)
        f.write(b)
        f.close()
        self.put(f.name,p_path=p_path,**kws)
        os.unlink(f.name)
    
    def put(self, file_path, p_path=None, description=None, block_size=DEFAULT_BLOCK_SIZE):
        if not p_path:
            p_path = file_path
        
        with open(file_path, "rb") as f:
            file_size = os.path.getsize(file_path)
            file_created = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

            parts = []
            start = 0
            while start < file_size:
                end = min(start + block_size, file_size)
                data = f.read(block_size)
                data_uuid = str(uuid.uuid4())
                parts.append({'uuid': data_uuid, 'start': start, 'end': end})
                cur = self.conn.cursor()
                cur.execute('INSERT INTO datas (uuid, data, path, start, end) VALUES (?, ?, ?, ?, ?)',
                            (data_uuid, data, p_path, start, end))
                start = end

            parts_json = json.dumps(parts)
            try:
                cur = self.conn.cursor()
                mt, ec = mimetypes.guess_type(file_path)
                
                cur.execute('''INSERT INTO files (path, created, modified, length, encoding, mimetype, description, parts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                            (p_path, file_created, file_modified, file_size, ec, mt, description, parts_json))
            except sqlite3.IntegrityError:
                cur.execute('DELETE FROM files WHERE path=?', (p_path,))
                cur.execute('''INSERT INTO files (path, created, modified, length, encoding, mimetype, description, parts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                            (p_path, file_created, file_modified, file_size, ec, mt, description, parts_json))
            self.conn.commit()

    def update_files_table(self, path, **fields):
        cur = self.conn.cursor()
        query = "UPDATE files SET "
        query += ', '.join([f"{k} = ?" for k in fields.keys()])
        query += " WHERE path = ?"
        cur.execute(query, (*fields.values(), path))
        self.conn.commit()

    def search(self, search_string):
        cur = self.conn.cursor()
        cur.execute('SELECT path FROM files WHERE path LIKE ?', (search_string ,))
        while True:
            result=cur.fetchone()
            if not result or result is None: break
            yield result[0]
        cur.close()
        #return [row['path'] for row in cur.fetchall()]
    
    def keys(self):
        cur = self.conn.cursor()
        cur.execute('SELECT path FROM files')
        while True:
            result=cur.fetchone()
            if not result or result is None: break
            yield result[0]
        cur.close()
    
    
    
    
    def delete(self, file_path):
        cur = self.conn.cursor()
        cur.execute('SELECT parts FROM files WHERE path=?', (file_path,))
        parts = cur.fetchone()
        if parts:
            for part in json.loads(parts['parts']):
                cur.execute('DELETE FROM datas WHERE uuid=?', (part['uuid'],))
            cur.execute('DELETE FROM files WHERE path=?', (file_path,))
            self.conn.commit
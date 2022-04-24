import json
import socket
import pandas as pd
from datetime import timedelta


class GraphData:
    def __init__(self):
        self.data = pd.DataFrame([], columns=['timestamp', 'value', 'forecast'])

    def append(self, elem):
        if elem['type'] == 'text':
            return
        if elem['type'] == 'series':
            elem['timestamp'] = pd.to_datetime(elem['timestamp'])
            del elem['type']
            # TODO: проверить, что дата позднее всех остальных

            elem = pd.Series(elem)
            if self.data.shape[0] > 0:
                self.data.drop(self.data.tail(1).index, inplace=True)
            # новый известный элемент
            elem['forecast'] = 0
            self.data = self.data.append(elem, ignore_index=True)
            # начало линии предикта
            elem['forecast'] = 1
            self.data = self.data.append(elem, ignore_index=True)
            # сам предикт
            elem['forecast'] = 1
            elem['timestamp'] += timedelta(days=1)
            self.data = self.data.append(elem, ignore_index=True).reset_index(drop=True)

    def get(self):
        return self.data


class Server:
    def __init__(self):
        #self.serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, proto=0)
        #self.serv_sock.bind(('127.0.0.1', 9999))
        #self.serv_sock.listen(10)
        self.data = GraphData()

    def get_start(self):
        return self.data.get()

    def get(self):
        client_sock, client_addr = self.serv_sock.accept()
        print('Connected by', client_addr)

        all_data = ''
        while True:
            data = client_sock.recv(4096).decode() #str(client_sock.recv(4096), 'utf-8')
            if not data:
                break
            all_data += data
            client_sock.sendall( f'ok: {len(data)} read'.encode() )

        elem = json.loads(all_data)
        print('Received:', elem)
        client_sock.close()
        self.data.append(elem)

        return self.data.get()

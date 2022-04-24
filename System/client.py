import socket
import pandas as pd
import time

if __name__ == '__main__':

    df = pd.DataFrame({
        "timestamp": [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-02'), pd.to_datetime('2021-01-03'),
              pd.to_datetime('2021-01-04'), pd.to_datetime('2021-01-05'),
                 pd.to_datetime('2021-01-05'), pd.to_datetime('2021-01-06')],
        "value": [1, 3, 4, 8, 6, 6, 7],
        #"forecast": [0, 0, 0, 0, 0, 1, 1]
    })
    df['timestamp'] = df['timestamp'].apply(str)
    df['type'] = 'series'

    buf_size = 1024
    for i, row in df.iterrows():
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(('127.0.0.1', 53220))

        raw_data = row.to_json().encode()
        for chunk_i in range(0, len(raw_data), buf_size):
            to_send = raw_data[chunk_i : chunk_i + buf_size]
            client_sock.sendall(to_send)
            resp = client_sock.recv(1024).decode() #str(client_sock.recv(1024), 'utf-8')
            if not resp or 'ok' not in resp:
                client_sock.close()
                raise ValueError(f'resp is {resp}')
            else:
                print(resp)

        client_sock.close()
        print(f'Sent row {i}', flush=True)
        time.sleep(20)
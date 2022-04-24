from backend import BackEnd

service = BackEnd()

if __name__ == '__main__':
    while True:
        service.step()

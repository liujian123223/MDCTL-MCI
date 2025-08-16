import argparse
import pickle
import socket
import time
import struct
import torch
from types import SimpleNamespace
from models.MDCTL_MCI import Model


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

def parse_args():
    parser = argparse.ArgumentParser(description='Collaborative training')
    parser.add_argument('--d_model', type=int, required=True, help='Model dimension')
    parser.add_argument('--d_ff', type=int, required=True, help='Feed-forward dimension')
    parser.add_argument('--ICB_hidden', type=int, required=True, help='ICB hidden dimension')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer learning rate')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--train_epochs', type=int, default=10, help='Train epochs')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=7002, help='Port number')
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients per epoch')
    parser.add_argument('--mask_rate_impute', type=float, default=0, help='Mask ratio')
    parser.add_argument('--seq_len', type=int, default=48, help='Input sequence length')
    return parser.parse_args()

def average_parameters(*args):
    result = args[0].clone()
    for i in range(1, len(args)):
        result += args[i]
    result /= len(args)
    for i in range(len(args)):
        args[i].copy_(result)


def socket_udp_server(d_model, d_ff, ICB_hidden, learning_rate, e_layers, seq_len, mask_rate_impute, train_epochs, host,
                      port, num_clients):
    configs = SimpleNamespace(
        d_model=d_model,
        d_ff=d_ff,
        ICB_hidden=ICB_hidden,
        e_layers=e_layers,
        seq_len=seq_len,
        task_name='long_term_forecast',
        pred_len=24,
        enc_in=10,
        subsequence_num=2,
        dropout=0.1
    )

    global_model = Model(configs)
    global_state_dict = global_model.state_dict()  # 获取初始全局模型的 state_dict

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(num_clients)
    log(f'Center Server started with d_model={d_model}, d_ff={d_ff}, ICB_hidden={ICB_hidden}, learning_rate={learning_rate}, e_layers={e_layers}, seq_len={seq_len}, mask={mask_rate_impute}, listening on {host}:{port}')
    for epoch in range(1, train_epochs + 1):
        log(f"Epoch {epoch}: Waiting for client connections")
        connected_socks = []
        res = []
        while len(connected_socks) < num_clients:
            try:
                s.settimeout(100)
                sock, addr = s.accept()
                log(f'Connection accepted from {addr}')
                data_length_bytes = sock.recv(4)
                if not data_length_bytes:
                    raise ValueError("No data length received")
                data_length = int.from_bytes(data_length_bytes, byteorder='big')
                received_data = b''
                while len(received_data) < data_length:
                    packet = sock.recv(data_length - len(received_data))
                    if not packet:
                        raise ConnectionError("Connection interrupted")
                    received_data += packet

                tmp = pickle.loads(received_data)
                log('Received data from client.')
                if tmp['num'] == epoch:
                    connected_socks.append(sock)
                    res.append(tmp['model'])

            except socket.timeout:
                log("Timeout waiting for client connections.")
                break
            except Exception as e:
                log(f"Error receiving data: {e}")

        if res:
            log(f"Epoch {epoch}: Received parameters from {len(res)} clients")
            log("Aggregating parameters...")

            for key in res[0].keys():
                client_params = [client_model[key] for client_model in res]
                average_parameters(*client_params)

            # 更新全局模型状态字典
            global_state_dict = res[0]
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

            # 准备发送聚合后的模型回客户端
            data = {'num': epoch, 'model': res[0]}
            log(f"Epoch {epoch}: Aggregation complete, sending back to clients")
            data = pickle.dumps(data)

            for sock in connected_socks:
                try:
                    data_length = len(data)
                    packed_length = struct.pack('!I', data_length)
                    sock.sendall(packed_length)
                    sock.sendall(data)
                except Exception as e:
                    log(f"Error sending data: {e}")
                finally:
                    sock.close()

            global_model = global_state_dict.copy()

        connected_socks.clear()

    final_global_model_path = f'final_global_model_{current_time}.pth'
    torch.save(global_model, final_global_model_path)
    log(f"Final global model saved to {final_global_model_path}")
    s.close()
    log('Training complete. Server shutting down.')


def main():
    args = parse_args()
    log(f"Server starting with parameters: d_model={args.d_model}, d_ff={args.d_ff}, ICB_hidden={args.ICB_hidden}, learning_rate={args.learning_rate}, e_layers={args.e_layers}, seq_len={args.seq_len}, mask={args.mask_rate_impute}, train_epochs={args.train_epochs}, host={args.host}, port={args.port}, num_clients={args.num_clients}")

    socket_udp_server(
        d_model=args.d_model,
        d_ff=args.d_ff,
        ICB_hidden=args.ICB_hidden,
        learning_rate=args.learning_rate,
        e_layers=args.e_layers,
        seq_len=args.seq_len,
        mask_rate_impute=args.mask_rate_impute,
        train_epochs=args.train_epochs,
        host=args.host,
        port=args.port,
        num_clients=args.num_clients
    )
if __name__ == '__main__':
    main()

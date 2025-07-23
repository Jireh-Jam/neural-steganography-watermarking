import reedsolo
def text_to_bits(text, length=128*128):
    bits = ''.join(format(ord(c), '08b') for c in text)
    bits = bits.ljust(length, '0')[:length]
    return np.array(list(bits), dtype=np.float32).reshape((128, 128, 1))

def bits_to_text(bit_array, threshold=0.6):
    """Convert numpy array of bits to text string"""
    # Convert bits to string of 0s and 1s
    # bits = ''.join(['1' if b > 0.5 else '0' for b in bit_array.flatten()])
    bits = ''.join(['1' if b > threshold else '0' for b in bit_array.flatten()])
    # Split into 8-bit chunks and convert to characters
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars).strip('\x00')

def encode_message(msg):
    rs = reedsolo.RSCodec(10)  # Can correct 5 byte errors
    return rs.encode(msg.encode())

def decode_message(encoded):
    rs = reedsolo.RSCodec(10)
    return rs.decode(encoded)[0].decode()



def q2a(test_string):
  encoding_types = ["utf-8", "utf-16", "utf-32"]
  print(f"{test_string=}")
  for encoding in encoding_types:
    encoded = test_string.encode(encoding)
    print(f"{encoding=}: {encoded}")
  
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])

def q2b(test_string: str):
  encoded = test_string.encode("utf-8")
  print(f"{test_string=}, {encoded=}")
  decoded = decode_utf8_bytes_to_str_wrong(encoded)
  print(f"{decoded=}")

def q3b():
  for i in range(0xff):
    for j in range(0xff):
      b = bytes([i, j])
      try:
        decoded = b.decode('utf-16')
        # print(f"{b=}, {decoded=}")
      except:
        print(f"failed to decode: {b}")
        return

test_string = "Hello 你好!"
q2a(test_string)

q2b("hello")
# q2b(test_string)

q3b()

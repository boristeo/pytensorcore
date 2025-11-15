new = """
    s_a_frag[0] = s_a[((((i%2)*256)+((tid/4)*16))+((tid%4)*2))];
    s_a_frag[1] = s_a[(((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+1)];
    s_a_frag[2] = s_a[(((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+8)];
    s_a_frag[3] = s_a[((((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+8)+1)];
    s_a_frag[4] = s_a[(((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+128)];
    s_a_frag[5] = s_a[((((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+128)+1)];
    s_a_frag[6] = s_a[((((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+128)+8)];
    s_a_frag[7] = s_a[(((((((i%2)*256)+((tid/4)*16))+((tid%4)*2))+128)+8)+1)];
"""
old = """
    s_a_frag[0] = s_a[(i%2)*256+(by)*0+(0+(tid/4))*16+(i)*0+(0+(tid%4)*2)*1];
    s_a_frag[1] = s_a[(i%2)*256+(by)*0+(0+(tid/4))*16+(i)*0+(1+(tid%4)*2)*1];
    s_a_frag[2] = s_a[(i%2)*256+(by)*0+(8+(tid/4))*16+(i)*0+(0+(tid%4)*2)*1];
    s_a_frag[3] = s_a[(i%2)*256+(by)*0+(8+(tid/4))*16+(i)*0+(1+(tid%4)*2)*1];
    s_a_frag[4] = s_a[(i%2)*256+(by)*0+(0+(tid/4))*16+(i)*0+(8+(tid%4)*2)*1];
    s_a_frag[5] = s_a[(i%2)*256+(by)*0+(0+(tid/4))*16+(i)*0+(9+(tid%4)*2)*1];
    s_a_frag[6] = s_a[(i%2)*256+(by)*0+(8+(tid/4))*16+(i)*0+(8+(tid%4)*2)*1];
    s_a_frag[7] = s_a[(i%2)*256+(by)*0+(8+(tid/4))*16+(i)*0+(9+(tid%4)*2)*1];
"""
def extract(code):
  return [line[line.index('[', line.index('='))+1:line.index(']', line.index('='))].replace('/','//') for line in code.split('\n') if line.strip()]


def extract(code):
  return [line.replace('/','//') for line in code.split('\n') if line.strip()]

def sizeof(t):
  return {'half': 2}[t]

i = 100
nexti = i+1
tid = 8
bx = 150
by = 100
M = 4096
N = 4096
K = 4096
Mmma = 16
Nmma = 8
Kmma = 16
half='half'

new = """
      (((((tid/2)*16)+((tid%2)*8))+(((i+1)%2)*256)))*sizeof(half)
      (((((tid/2)*4096)+((tid%2)*8))+(by*65536))+((i+1)*16))
"""
old = f"""
      (nexti%2*{Mmma*Kmma}+tid*8)*sizeof(half)
      by*{Mmma*K}+nexti*{Kmma}+(tid/2)*{K}+(tid%2)*8
"""

for old, new in zip(extract(old), extract(new)):
  print('old', eval(old), old)
  print('new', eval(new), new)
  print()

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

i = 0
tid = 1
bx = 0
by = 0



for old, new in zip(extract(old), extract(new)):
  print(eval(old), old)
  print(eval(new), new)
  print()

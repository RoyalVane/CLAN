f = open('PPT5.txt', 'w+')
for i in range(0, 300):
    f.write('stuttgart_02_000000_{0:06d}_leftImg8bit.png\n'.format(i+5100))

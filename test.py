import sys

def main(lines):
    # このコードは標準入力と標準出力を用いたサンプルコードです。
    # このコードは好きなように編集・削除してもらって構いません。
    # ---
    # This is a sample code to use stdin and stdout.
    # Edit and remove this code as you like.
    lst = []
    for i, v in enumerate(lines):
        v = int(v)
        while v > 0:
            lst.append(int(v)%10)
            v //= 10 
        
        lst_reverse =sorted(lst)

        for i, v in enumerate(lst_reverse):
            
            if(v != 0):

                non_zero_index = i
                non_zero_valid = True
                break

        if non_zero_valid:
            tmp = lst_reverse[0]
            lst_reverse[0] = lst_reverse[i]
            lst_reverse[i] = tmp
        print(int("".join(map(str, lst_reverse))))


            
        


if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)

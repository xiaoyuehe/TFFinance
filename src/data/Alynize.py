
if __name__ == '__main__':
    df = [1,2,3,5,7,6,4,6,8,9]
    mid = []
    cur = []
    cur_dir = 1
    cur_value = df[0]
    cur.append(cur_value)
    mid.append(cur)
    for i in range(len(df)-1):
        new_value = df[i+1]
        new_dir = 1 if new_value >= cur_value else -1

        cur_value = new_value
        if new_dir * cur_dir < 0:
            cur = []
            mid.append(cur)

        cur_dir = new_dir
        cur.append(cur_value)

    for i in range(len(mid)):
        print("-"*20)
        for j in range(len(mid[i])):
            print(mid[i][j])


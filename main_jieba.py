import time
t1=time.time()
import jieba

def count_duplicate_elements(list1, list2):
    duplicate_count = {}
    for item in list1:
        if item in list2:
            duplicate_count[item] = duplicate_count.get(item, 0) + 1
    return len(duplicate_count)

#print(count_duplicate_elements(list1, list2))
file_f=[]
file_s=[]
with open('任务.txt', encoding='gbk') as f:
    file = f.readlines()  # 打开任务并按行读取生成列表
symbol=' '
list1=('很美','更美','开心','惬意','的美','惊喜','无聊','热闹','真好','美好','漂亮','太美','非常','冰','雪','姐','泳','想象','水域','壮观','好看','愉快','真美','喜欢','震撼','舒服','门票','最美','挺美','价格','表哥','闺蜜','贵','冬','爷','船','趣','感觉','提醒'
       '男友','女友','情人','便宜','小贵','陶醉','一起','价','鲁冰花','妈','大','乐','行','静','猫','爱人','浪漫','喊','播放','园区'
       '友','不错','家乡','美丽','老婆',"老公","迷人","便利",'喜欢','环','好多','电视','这么','美景','鸳鸯','鸭','挺多','喇叭','两元'
       ,"过瘾",'看到','很香',"花香",'美艳','雪糕','湖','拍','疫情','预约','船','风','工作人员','停车','15元','汗','运动','汉','落日'
       '只能','超','孩','团','园内','很多','不少','鸟','摄','不多','野','春','确实','带着','好奇','广播','鱼','欢乐','自然','2元'
       '收费','提醒','售卖','河','人山','人海','人挨','人挤','人多','人流','提醒','拥挤','花海','树木','花草','开放','吹','阿姨',''
       ) #地标 木樨地 电视塔 解放军总医院 钓鱼台 国宾馆 白堆子地铁 公主坟 西门 停车场 大中电器 京港澳高速 阜石
set2=[]
times  = 0
times1 = 0
for strs in file:
    file_f = jieba.cut(file[0])
    for word in list1:
        if strs.find(word) != -1 and 4<=len(strs):
            #print(strs)
            times+=1
            if count_duplicate_elements(file_f, file_s) >= 1 :
                strs = '^'.join(file_f + file_s)
                file_s = file_f
                times1 += 1
            set2.append(strs)
        continue
#print(set2)
set3=set(set2)
#print(set3)
print('数据量',times-1)
print('筛选量',len(set3),end='')
print("合并量",times1 )
with open('newfile.csv','w+',encoding='gbk')as h:
    h.writelines(set3)
t2=time.time()
print('\n',t2 - t1)
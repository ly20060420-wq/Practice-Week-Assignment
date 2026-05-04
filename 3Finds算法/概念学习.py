attributes = [ ['Sunny', 'Rainy'], 
              ['Warm', 'Cold'], 
              ['Normal', 'High'], 
              ['Strong', 'Light'], 
              ['Warm', 'Cool'], 
              ['Same', 'Change']]

examples = [[['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], True],
            [['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True],
            [['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False],
            [['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True]]

#功能：判断假设 h 是否与样例 <e, elabel> 一致
#h(list):当前假设, e(list):样例输入
def agree(h,e,elabel):
    cover=True#默认假设覆盖样例
    #正例：如果h能覆盖e说明h预测e的标签与真实标签相符，负例：h不能覆盖e说明h预测与真实标签相符
    for i in range(len(h)):
        if(h[i]=='$'):
            cover=False
            break
        if(h[i]=='?'):
            continue
        if(h[i]!=e[i]):
            cover=False
            break
    if elabel:
        return cover
    else:
        return not cover


def find_S(examples): 
    h = ['$']*len(attributes) # 初始化为全空的最特殊假设 
    for e in examples: 
        if not e[1]: # 跳过负例
            continue
        if not agree(h,e[0],True):#当前假设不覆盖正例e
            h=min_generalize(h,e[0])#将h极小一般化以覆盖e
    return h



def is_special_than(h1,h2,strict=False):
    all_=True#h1[i]特殊于或等于h2[i]
    any_=False#h1[i]严格特殊于h2[i]
    for i in range(len(h1)):
        if(h2[i]=='?'):
            continue
        if(h1[i]=='$'):
            continue
        if(h1[i]!=h2[i]):#h2是具体值，h1不是$也不等于h2,则h1不可能比h2特殊
            all_=False
            break
    #strict: 是否严格比较（若为True，则h1h2完全相同返回False,即还需至少有一个属性h2是'?'，h1是具体值或'$' / h2是具体值，h1 是'$'）
    if strict:
        for i in range(len(h1)):
            if(h2[i]=='?' and h1[i]!='?') or (h2[i]!='$' and h1[i]=='$'):
                any_=True
                break
        return any_ and all_
    else:
        return all_


def is_general_than(h1,h2,strict=False):
    return is_special_than(h2,h1,strict=strict)

#功能：将假设h极小一般化以覆盖正例 e
def min_generalize(h,e):
    new=[]
    for i in range(len(h)):
        if(h[i]=='$' or h[i]==e[i]):
            new.append(e[i])
        else:
            new.append('?')
    return new
        
#功能：生成假设 h 的极小特殊化候选集
def min_specialize(h):
    special=[]
    for i in range(len(h)):
        if(h[i]=='?'):
            for value in attributes[i]:
                #将第 i 个属性替换为具体值
                new=h.copy()
                new[i]=value
                special.append(new)#把所有可能的特化分支全部列出来放在列表中
    return special




# 功能：实现候选消除算法
def generate_SG(examples):

    empty_h=['$']*len(attributes)
    full_h=['?']*len(attributes)
    S,G=[empty_h],[full_h]    
    #维护S集合(最特殊边界)和G集合(最一般边界)
    #S边界：从最特殊假设出发，遇到正例向一般方向移动
    #G边界：从最一般假设出发，遇到反例向特殊方向移动

    print("\n候选消除算法开始")
    print(f"初始 S = {S}")
    print(f"初始 G = {G}")
    for idx, e in enumerate(examples):
        x, label = e[0], e[1]
        print(f"\n处理样例{idx + 1}: [{x}, {label}]")
        if e[1]:#正例
            G=[h for h in G if agree(h,e[0],True)]#删除G中不覆盖e的假设
            S=[min_generalize(s,e[0]) for s in S]#将S中每个假设s极小一般化以覆盖e
            S=[s for s in S if any(is_general_than(g,s) for g in G)]#检查新假设是否在变型空间内（被G中某个假设涵盖）
            S=[s for s in S if not any(is_general_than(s, other, strict=True) for other in S)]
        else:#反例
            S=[s for s in S if agree(s,e[0],False)]#删除S中覆盖e的假设
            #特化G以排除反例
            new_G=[]
            for g in G:
                if agree(g,e[0],False):#如果g不覆盖反例e，则g仍然是有效的假设
                    new_G.append(g)
                else:#如果g覆盖反例e，则需要特化g以排除e
                    new_G.extend(min_specialize(g))#将g极小特化,extend()方法将min_specialize(g)返回的列表中的元素逐个添加到new_G中
            
            new_G = [g for g in new_G if any(is_general_than(g, s) for s in S) and agree(g, e[0], False)]
            G = [g for g in new_G if not any(is_special_than(g, other, strict=True) for other in new_G)]
        if not S or not G:
            print(f"在处理样例 {idx + 1} 时发现冲突！可能存在噪声数据")
            break
        print(f" S = {S}")
        print(f" G = {G}")
    return S,G


#功能：生成变型空间（Version Space），即所有与数据一致的假设.BFS
def generate_VS(S,G):
    VS=[]

    queue=[tuple(s) for s in S]#将list转为tuple,以便存入set去重、0\1查找
    #队列初始化：S集合中所有元素作为BFS起点
    visited=set(queue)
    #每次从队列弹出一个假设，对其“极小一般化”,每次仅将其中一个具体属性替换为'?'，生成下一层候选假设
    while queue:
        current_h_tuple=queue.pop(0)
        current_h=list(current_h_tuple)

        VS.append(current_h)#只要能进队列，一定被G涵盖
                                                                                                  
        for i in range(len(current_h)):
            if current_h[i]!='?':
                next_h=current_h.copy()
                next_h[i]='?'#将第 i 个属性泛化为 '?'
                next_h_tuple=tuple(next_h)

                if next_h_tuple not in visited:
                    if any(is_special_than(next_h,g) for g in G):
                        visited.add(next_h_tuple)
                        queue.append(next_h_tuple)
    return VS



if __name__=='__main__':
    S=find_S(examples)
    print(f"\nFind_S输出：\n{S}")

    S,G=generate_SG(examples)
    print(f"\n候选消除算法输出：\nS：{S}\nG：{G}\n")
    VS=generate_VS(S,G)
    print(f"变型空间为：")
    for h in VS:
        print(f"{h}")

    print("\n扩展任务：添加新训练样例后的变型空间收缩")
    examples_extended = examples + [
        [['Sunny', 'Cold', 'Normal', 'Strong', 'Warm', 'Same'], False]
    ]
    S2, G2 = generate_SG(examples_extended)
    print(f"\n扩展后 S = {S2}")
    print(f"扩展后 G = {G2}")

    VS2 = generate_VS(S2, G2)
    print("\n扩展后的变型空间：")
    for h in VS2:
        print(f"  {h}")
    print(f"变型空间大小: {len(VS2)}（收缩前为 {len(VS)}）")
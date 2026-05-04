# ============================================================
# 机器学习实验：概念学习 —— Find-S 算法 与 候选消除算法
# ============================================================
# 背景：EnjoySport 问题
#   目标概念：Aldo 进行水上运动的日子
#   每个实例由 6 个属性描述（Sky, AirTemp, Humidity, Wind, Water, Forecast）
#   假设用属性约束的合取式表示，每个约束可以是：
#     - 具体属性值（如 'Sunny'）：只接受该值
#     - '?'：接受任意值（最一般）
#     - '$'：不接受任何值（最特殊，相当于空假设 ∅）
# ============================================================

# 属性的所有可能取值（每个属性有 2 个可能的值）
attributes = [['Sunny', 'Rainy'],      # Sky（天空）
              ['Warm', 'Cold'],         # AirTemp（气温）
              ['Normal', 'High'],       # Humidity（湿度）
              ['Strong', 'Light'],      # Wind（风力）
              ['Warm', 'Cool'],         # Water（水温）
              ['Same', 'Change']]       # Forecast（预报）

# 训练样例：每个样例格式为 [属性值列表, 标签(True=正例, False=反例)]
examples = [
    [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], True],   # 正例1
    [['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True],     # 正例2
    [['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False],  # 反例3
    [['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True]    # 正例4
]


# ============================================================
# 函数1：agree(h, e, elabel)
# ============================================================
# 功能：判断假设 h 是否与样例 <e, elabel> 一致
#
# 运行逻辑：
#   1. 首先判断 h 是否"覆盖"（cover）样例 e：
#      - 逐个比较 h 和 e 的每个属性：
#        · 如果 h[i] == '$'（空约束），说明 h 不接受任何值 → 不覆盖
#        · 如果 h[i] == '?'（任意约束），说明该属性接受任意值 → 跳过继续
#        · 如果 h[i] != e[i]（具体值不匹配） → 不覆盖
#      - 如果所有属性都通过检查，则 h 覆盖 e（cover=True）
#   2. 然后根据标签判断"一致性"：
#      - 如果是正例（elabel=True）：h 覆盖 e 才一致
#      - 如果是反例（elabel=False）：h 不覆盖 e 才一致
#
# 参数：
#   h (list): 当前假设，如 ['Sunny', '?', 'Normal', ...]
#   e (list): 样例输入，如 ['Sunny', 'Warm', ...]
#   elabel (bool): 样例标签，True=正例，False=反例
# 返回：
#   bool: 假设 h 与样例 <e, elabel> 是否一致
# ============================================================
def agree(h, e, elabel):
    cover = True  # 默认假设 h 覆盖样例 e
    for i in range(len(h)):
        if h[i] == '$':
            # '$' 表示不接受任何值，遇到就说明 h 不可能覆盖任何样例
            cover = False
            break
        if h[i] == '?':
            # '?' 表示接受任意值，该属性一定匹配，跳过
            continue
        if h[i] != e[i]:
            # 具体属性值不同，不覆盖
            cover = False
            break
    # 一致性判断：
    #   正例要求 h 覆盖 e（cover=True 时一致）
    #   反例要求 h 不覆盖 e（cover=False 时一致）
    if elabel:
        return cover
    else:
        return not cover


# ============================================================
# 函数2：is_special_than(h1, h2, strict=False)
# ============================================================
# 功能：判断假设 h1 是否比 h2 更特殊（more specific）
#
# 运行逻辑：
#   "h1 比 h2 更特殊" 意味着 h1 覆盖的实例集合 ⊆ h2 覆盖的实例集合
#   在合取式表示中，这等价于：
#     对于每个属性 i：
#       - 如果 h2[i] == '?'，则 h1[i] 可以是任何值（'?' 最一般，什么都行）
#       - 否则 h1[i] 必须等于 h2[i]（h1 的约束不能比 h2 更宽松）
#
#   strict 参数控制是否严格比较：
#     - strict=False（默认）：h1 等于 h2 也返回 True（≤_s 关系）
#     - strict=True：h1 必须严格比 h2 更特殊（<_s 关系），
#       即至少有一个属性 h2[i]='?' 而 h1[i]!='?'
#
# 注意：原代码在 strict 分支有缩进 bug（return 写在 for 循环内），已修复
# ============================================================
def is_special_than(h1, h2, strict=False):
    all_ = True   # h1[i] 是否在每个属性上都 "特殊于或等于" h2[i]
    any_ = False  # 是否存在至少一个属性使得 h1[i] 严格特殊于 h2[i]

    # 第一步：检查 h1 是否在所有属性上都不比 h2 更一般
    for i in range(len(h1)):
        if h2[i] == '?':
            # h2 在该属性上接受任意值，所以 h1 无论是什么都满足 "特殊于或等于"
            continue
        if h1[i] != h2[i]:
            # h2[i] 是具体值，而 h1[i] 不同（可能是 '?' 或另一个具体值）
            # → h1 不可能比 h2 更特殊
            all_ = False
            break

    if strict:
        # 第二步（严格模式）：还需要至少存在一个属性使 h1 严格更特殊
        # 即 h2[i]='?' 而 h1[i]!='?'
        for i in range(len(h1)):
            if h2[i] == '?' and h1[i] != '?':
                any_ = True
                break
        # 【原代码 bug 修复】：这里的 return 应该在 for 循环外面
        return any_ and all_
    else:
        return all_


# ============================================================
# 函数3：is_general_than(h1, h2, strict=False)
# ============================================================
# 功能：判断假设 h1 是否比 h2 更一般（more general）
#
# 运行逻辑：
#   "h1 比 h2 更一般" 等价于 "h2 比 h1 更特殊"
#   所以直接调用 is_special_than，交换参数即可
#
# 注意：原代码 bug —— 调用 is_special_than 时 strict 参数写死了 False，已修复
# ============================================================
def is_general_than(h1, h2, strict=False):
    # 【原代码 bug 修复】：应传递实际的 strict 参数，而非固定 False
    return is_special_than(h2, h1, strict=strict)


# ============================================================
# 函数4：min_generalize(h, e)
# ============================================================
# 功能：将假设 h 极小一般化（minimal generalization）以覆盖正例 e
#
# 运行逻辑（逐属性处理）：
#   对于每个属性 i：
#     - 如果 h[i] == '$'（空值）：
#       这是 h 第一次接触该属性，直接用 e[i] 的值替换 '$'
#       例如：h=['$','$',...], e=['Sunny','Warm',...]
#       → new[i] = 'Sunny'（从空假设开始，直接取正例的值）
#
#     - 如果 h[i] == e[i]（假设和样例在该属性上一致）：
#       无需泛化，保持不变
#       例如：h[i]='Sunny', e[i]='Sunny' → new[i]='Sunny'
#
#     - 如果 h[i] != e[i] 且 h[i] != '$'：
#       假设和样例在该属性上冲突，需要泛化为 '?'（接受任意值）
#       例如：h[i]='Normal', e[i]='High' → new[i]='?'
#       这是"极小"一般化，因为 '?' 是最小的能同时覆盖两者的约束
#
#     - 如果 h[i] == '?'：
#       已经是最一般的了，保持不变
#
# 注意：原代码有 bug —— 缺少 h[i]==e[i] 的情况处理，且用了 if 而非 elif，已修复
# ============================================================
def min_generalize(h, e):
    new = []
    for i in range(len(h)):
        if h[i] == '$':
            # 空约束 → 直接采用正例的属性值
            new.append(e[i])
        elif h[i] == '?' or h[i] == e[i]:
            # 已经是 '?' 或者值相同 → 保持不变
            new.append(h[i])
        else:
            # 值不同 → 泛化为 '?'（极小一般化）
            new.append('?')
    return new


# ============================================================
# 函数5：min_specialize(h)
# ============================================================
# 功能：生成假设 h 的所有极小特殊化（minimal specializations）候选集
#
# 运行逻辑：
#   极小特殊化是候选消除算法中处理反例时的关键操作。
#   当 G 集合中的某个假设 g 覆盖了一个反例时，需要将 g 特殊化，
#   使其不再覆盖该反例，同时尽量保持一般性。
#
#   方法：遍历每个属性 i，如果 h[i] == '?'（最一般约束），
#         则可以将其替换为该属性的每一个具体取值。
#         每次只替换一个属性（极小特殊化 = 一次只收紧一个约束）。
#
#   例如：h = ['?', 'Warm', '?', '?', '?', '?']
#   对于属性0（可取 'Sunny' 或 'Rainy'），生成：
#     ['Sunny', 'Warm', '?', '?', '?', '?']
#     ['Rainy', 'Warm', '?', '?', '?', '?']
#   对于属性1（已经是 'Warm'，不是 '?'），跳过
#   对于属性2（可取 'Normal' 或 'High'），生成：
#     ['?', 'Warm', 'Normal', '?', '?', '?']
#     ['?', 'Warm', 'High', '?', '?', '?']
#   ... 以此类推
#
#   使用 yield（生成器），按需生成候选假设，节省内存
# ============================================================
def min_specialize(h):
    for i in range(len(h)):
        if h[i] == '?':
            # 该属性当前是 '?'（接受任意值），可以特殊化为具体值
            for val in attributes[i]:
                # 复制一份假设，将第 i 个属性替换为具体值 val
                new = h.copy()
                new[i] = val
                yield new
        # 如果 h[i] 已经是具体值，则无法再进一步特殊化（'$' 太极端，不考虑）


# ============================================================
# 函数6：find_S(examples)
# ============================================================
# 功能：实现 Find-S 算法，找到与所有正例一致的最特殊假设
#
# 算法思想：
#   Find-S 利用假设空间的偏序结构（一般到特殊序），
#   从最特殊假设（全 '$'）出发，沿偏序链逐步向一般方向移动。
#
# 算法步骤：
#   1. 初始化 h 为最特殊假设 ['$', '$', '$', '$', '$', '$']
#      （不覆盖任何实例）
#   2. 遍历每个训练样例：
#      - 如果是反例（e[1]=False）：跳过（Find-S 只处理正例）
#      - 如果是正例（e[1]=True）：
#        检查当前假设 h 是否覆盖该正例
#        · 如果覆盖 → 无需修改
#        · 如果不覆盖 → 调用 min_generalize(h, e[0]) 极小一般化
#   3. 返回最终假设 h
#
# 特点：
#   - 只处理正例，忽略反例
#   - 输出 H 中与所有正例一致的最特殊假设
#   - 局限性：无法利用反例信息，且只输出一个假设
#
# 注意：原代码有两个 bug：
#   1. 判断条件 `not all(h[i] in (e[0][i], '$', '?') ...)` 缺少 '$'
#      → 导致第一个正例处理不正确（原代码条件中缺少 '$'，已在指导手册中修正）
#   2. return h 写在了 for 循环内部 → 只处理一个正例就返回了
#   已全部修复
# ============================================================
def find_S(examples):
    h = ['$'] * len(attributes)  # 步骤1：初始化为最特殊假设（全 $）
    print(f"初始假设 h = {h}")

    for idx, e in enumerate(examples):
        if not e[1]:
            # 跳过反例（Find-S 不处理反例）
            print(f"  样例{idx + 1} {e[0]} 是反例，跳过")
            continue

        # 当前样例是正例
        if not agree(h, e[0], True):
            # 假设 h 不覆盖该正例 → 需要泛化
            h = min_generalize(h, e[0])
            print(f"  样例{idx + 1} {e[0]} 是正例，h 不覆盖，泛化后 h = {h}")
        else:
            print(f"  样例{idx + 1} {e[0]} 是正例，h 已覆盖，无需修改")

    # 【原代码 bug 修复】：return 应该在 for 循环外面
    return h


# ============================================================
# 函数7：generate_SG(examples)
# ============================================================
# 功能：实现候选消除算法（Candidate Elimination Algorithm）
#       维护 S 集合（最特殊边界）和 G 集合（最一般边界）
#
# 算法思想：
#   候选消除算法同时从两个方向逼近目标概念：
#   - S 边界：从最特殊假设出发，遇到正例时向一般方向移动
#   - G 边界：从最一般假设出发，遇到反例时向特殊方向移动
#   两个边界之间的所有假设构成"变型空间"（Version Space）
#
# 算法步骤：
#   1. 初始化：
#      S = {最特殊假设}，即 [['$','$','$','$','$','$']]
#      G = {最一般假设}，即 [['?','?','?','?','?','?']]
#
#   2. 对每个训练样例 d：
#      如果 d 是正例：
#        a) 从 G 中移去所有与 d 不一致的假设
#           （G 中的假设应该覆盖所有正例）
#        b) 对 S 中每个与 d 不一致的假设 s：
#           - 从 S 中移去 s
#           - 将 s 的极小一般化加入 S，要求：
#             · 新假设与 d 一致（覆盖该正例）
#             · 新假设被 G 中某个假设所涵盖（在变型空间内）
#           - 移去 S 中所有被其他 S 成员涵盖（更一般）的假设
#             （保持 S 是极大特殊的）
#
#      如果 d 是反例：
#        a) 从 S 中移去所有与 d 不一致的假设
#           （S 中的假设不应覆盖反例）
#        b) 对 G 中每个与 d 不一致的假设 g：
#           - 从 G 中移去 g
#           - 将 g 的极小特殊化加入 G，要求：
#             · 新假设与 d 一致（不覆盖该反例）
#             · 新假设涵盖 S 中某个假设（在变型空间内）
#           - 移去 G 中所有被其他 G 成员涵盖（更特殊）的假设
#             （保持 G 是极大一般的）
#
#   3. 返回最终的 S 和 G
# ============================================================
def generate_SG(examples):
    # 初始化
    empty_h = ['$'] * len(attributes)  # 最特殊假设（不覆盖任何实例）
    full_h = ['?'] * len(attributes)   # 最一般假设（覆盖所有实例）
    S = [empty_h]
    G = [full_h]

    print("=" * 60)
    print("候选消除算法开始")
    print(f"初始 S = {S}")
    print(f"初始 G = {G}")
    print("=" * 60)

    for idx, e in enumerate(examples):
        x, label = e[0], e[1]
        print(f"\n处理样例{idx + 1}: {x}, 标签={'正例' if label else '反例'}")

        if label:
            # ========== 正例处理 ==========
            # 步骤 a：从 G 中移去所有与正例不一致的假设
            # （G 中的假设必须覆盖正例，否则不一致）
            G = [g for g in G if agree(g, x, True)]

            # 步骤 b：泛化 S 以覆盖该正例
            new_S = []
            for s in S:
                if agree(s, x, True):
                    # s 已经覆盖该正例，保持不变
                    new_S.append(s)
                else:
                    # s 不覆盖该正例 → 极小一般化
                    s_new = min_generalize(s, x)
                    # 检查新假设是否在变型空间内（被 G 中某个假设涵盖）
                    if any(is_general_than(g, s_new) for g in G):
                        new_S.append(s_new)
            S = new_S

            # 移去 S 中被其他 S 成员涵盖（更一般）的假设
            # 即：如果 S 中存在 s1 比 s2 更一般（s1 涵盖 s2），则移去 s1
            # 因为 S 要保持"极大特殊"，所以要移去更一般的那个
            S = remove_more_general(S)

        else:
            # ========== 反例处理 ==========
            # 步骤 a：从 S 中移去所有与反例不一致的假设
            # （S 中的假设不应覆盖反例，如果覆盖了就不一致）
            S = [s for s in S if agree(s, x, False)]

            # 步骤 b：特化 G 以排除该反例
            new_G = []
            for g in G:
                if agree(g, x, False):
                    # g 不覆盖该反例，已经一致，保持不变
                    new_G.append(g)
                else:
                    # g 覆盖了该反例 → 需要极小特殊化
                    for g_new in min_specialize(g):
                        # 检查新假设是否：
                        # 1. 与该反例一致（不覆盖反例）
                        # 2. 在变型空间内（涵盖 S 中某个假设）
                        if agree(g_new, x, False) and \
                           any(is_special_than(s, g_new) for s in S):
                            new_G.append(g_new)
            G = new_G

            # 移去 G 中被其他 G 成员涵盖（更特殊）的假设
            # 因为 G 要保持"极大一般"，所以要移去更特殊的那个
            G = remove_more_special(G)

        print(f"  S = {S}")
        print(f"  G = {G}")

    return S, G


def remove_more_general(S):
    """
    从 S 集合中移去多余的（被其他成员涵盖的更一般的）假设。
    S 应保持极大特殊：如果 s1 比 s2 更一般（即 s2 更特殊），
    那么 s1 是多余的，应被移去，保留更特殊的 s2。
    """
    result = []
    for i, s in enumerate(S):
        # 检查是否存在其他假设比 s 严格更特殊
        # 如果有，说明 s 太一般了，不应该留在 S 中
        is_redundant = False
        for j, other in enumerate(S):
            if i != j and is_general_than(s, other, strict=True):
                # s 比 other 严格更一般 → s 是多余的
                is_redundant = True
                break
        if not is_redundant:
            result.append(s)
    return result


def remove_more_special(G):
    """
    从 G 集合中移去多余的（被其他成员涵盖的更特殊的）假设。
    G 应保持极大一般：如果 g1 比 g2 更特殊（即 g2 更一般），
    那么 g1 是多余的，应被移去，保留更一般的 g2。
    """
    result = []
    for i, g in enumerate(G):
        is_redundant = False
        for j, other in enumerate(G):
            if i != j and is_special_than(g, other, strict=True):
                # g 比 other 严格更特殊 → g 是多余的
                is_redundant = True
                break
        if not is_redundant:
            result.append(g)
    return result


# ============================================================
# 函数8：generate_VS(S, G)
# ============================================================
# 功能：根据 S 边界和 G 边界，枚举变型空间中的所有假设
#
# 运行逻辑：
#   变型空间 VS = {h ∈ H | (∃s ∈ S)(∃g ∈ G)(g ≥_g h ≥_g s)}
#   即 VS 中的假设 h 必须：
#     - 比 S 中某个假设更一般（或相等）
#     - 比 G 中某个假设更特殊（或相等）
#
#   方法：枚举假设空间 H 中所有可能的假设，筛选满足条件的
#   每个属性位置可以取：该属性的所有具体值 或 '?'
#   （不考虑 '$'，因为包含 '$' 的假设不覆盖任何实例，
#    除非 S 本身就是全 '$'）
#
#   使用递归或迭代生成所有可能的假设组合
# ============================================================
def generate_VS(S, G):
    """
    枚举变型空间中的所有假设。
    遍历所有可能的假设 h，检查 h 是否在 S 和 G 之间。
    """
    # 每个属性的候选值 = 该属性的所有具体值 + '?'
    all_values = [attr + ['?'] for attr in attributes]

    VS = []

    def enumerate_hypotheses(pos, current):
        """递归枚举所有可能的假设"""
        if pos == len(attributes):
            # 构造完一个完整假设，检查是否在变型空间内
            h = current.copy()
            # 条件：h 比 G 中某个假设更特殊（或相等），且比 S 中某个假设更一般（或相等）
            if any(is_special_than(h, g) for g in G) and \
               any(is_general_than(h, s) for s in S):
                VS.append(h)
            return

        for val in all_values[pos]:
            current.append(val)
            enumerate_hypotheses(pos + 1, current)
            current.pop()

    enumerate_hypotheses(0, [])
    return VS


# ============================================================
# 主程序：运行 Find-S 和候选消除算法
# ============================================================
if __name__ == '__main__':
    # ===== 任务1：Find-S 算法 =====
    print("=" * 60)
    print("任务1：Find-S 算法")
    print("=" * 60)
    result = find_S(examples)
    print(f"\nFind-S 最终输出: {result}")
    print(f"预期结果: ['Sunny', 'Warm', '?', 'Strong', '?', '?']")

    # ===== 任务2：候选消除算法 =====
    print("\n\n")
    S, G = generate_SG(examples)

    print("\n" + "=" * 60)
    print("候选消除算法最终结果")
    print("=" * 60)
    print(f"S 集合（最特殊边界）: {S}")
    print(f"G 集合（最一般边界）: {G}")

    # 生成并显示变型空间
    print("\n变型空间（Version Space）中的所有假设：")
    VS = generate_VS(S, G)
    for h in VS:
        print(f"  {h}")
    print(f"变型空间大小: {len(VS)}")

    # ===== 扩展任务3：添加新训练样例 =====
    print("\n\n")
    print("=" * 60)
    print("扩展任务：添加新训练样例后的变型空间收缩")
    print("=" * 60)
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
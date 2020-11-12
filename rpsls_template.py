
"""
第一个小项目：Rock-paper-scissors-lizard-Spock
作者：土木二班陈宇翔
日期：2020/11/12
"""

import random



# 0 - 石头
# 1 - 史波克
# 2 - 纸
# 3 - 蜥蜴
# 4 - 剪刀

# 以下为完成游戏所需要用到的自定义函数

def name_to_number(name):
    if name=="石头":
        a=0
    elif name=="史波克":
        a=1
    elif name=="纸":
        a=2
    elif name=="蜥蜴":
        a=3
    elif name=="剪刀":
        a=4
    else:a=5
    return a
    """
    将游戏对象对应到不同的整数
    """

def number_to_name(number):
    if number==0:
        comp_name="石头"
    elif number==1:
        comp_name="史波克"
    elif number==2:
        comp_name='纸'
    elif number==3:
        comp_name="蜥蜴"
    elif number==4:
        comp_name="剪刀"
    return comp_name
    """
    将整数 (0, 1, 2, 3, or 4)对应到游戏的不同对象
    """

    # 使用if/elif/else语句将不同的整数对应到游戏的不同对象
    # 不要忘记返回结果


def rpsls(player_choice):
    number=random.randint(0,4)
    print("----------------")
    print("您的选择是"+player_choice)
    print("计算机的选择是"+number_to_name(number))
    a=name_to_number(player_choice)
    if a==5:
        print("Error: No Correct Name")
    elif a==number:
        print("您和计算机出的一样呢")
    elif a<=2 and a-number<=-3:
        print("您赢了")
    elif a>2 and a-number<=2:
        print("您赢了")
    else:print("计算机赢了")

    """
    用户玩家任意给出一个选择，根据RPSLS游戏规则，在屏幕上输出对应的结果

    """

print("欢迎使用RPSLS游戏")
print("----------------")
print("请输入您的选择:")
choice_name=input()
rpsls(choice_name)
input()





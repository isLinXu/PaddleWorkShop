
import paddlehub as hub

'''
def generate_image(
          text_prompts:str,输入的语句，描述想要生成的图像的内容。通常比较有效的构造方式为 "一段描述性的文字内容" + "指定艺术家的名字"，
          如"in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."。
          prompt的构造可以参考网站。
          style: Optional[str] = "油画",'水彩','粉笔画','卡通','儿童画','蜡笔画','探索无限'
          topk: Optional[int] = 6,保存前多少张图，最多保存6张
          output_dir: Optional[str] = 'ernievilg_output')保存输出图像的目录，默认为"ernievilg_output"。

'''

module = hub.Module(name="ernie_vilg")
results = module.generate_image(
    text_prompts=["玫瑰工厂","黄金之国","黎明神殿"],
    style='卡通',
    topk=6,
    output_dir='./save4'
)

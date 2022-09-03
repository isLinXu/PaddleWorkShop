
import paddlehub as hub

model = hub.Module(name='plato-mini')
data = [["你是谁？"], ["你好啊。", "吃饭了吗？",]]
result = model.predict(data)
print('result:', result)
import sys
import time
def typewriter_effect(text, delay=0.1):
   for char in text:
       sys.stdout.write(char)
       sys.stdout.flush()
       time.sleep(delay)
   print()
print("我是DeepSeek，很高兴见到你")
while True:
   user_input = input("给DeepSeek发送消息").lower()
   if user_input == '拜拜':
       typewriter_effect("拜拜！祝你一切顺利，下次见啦～")
       break
   if user_input == '滚':
       typewriter_effect("滚！")
       break
   else:
       typewriter_effect("服务器繁忙，请稍后再试")
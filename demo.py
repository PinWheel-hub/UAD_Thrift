class MyClass:
    def __init__(self):
        self.a = 0
    def my_method(self):
        self.a += 1
        print("This is a method of MyClass", self.a)

def another_function():
    c = a
    print(c)
    b = my_object
    b.my_method()

if __name__ == '__main__':
    # 在主函数中创建一个对象
    a = 10
    my_object = MyClass()

    # 调用其他函数，将对象作为参数传递
    another_function()
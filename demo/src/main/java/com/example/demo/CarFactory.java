package com.example.demo;

// 接口Factory<T>的实现类，CarFactory<T>标明是个模版
public class CarFactory<T> implements Factory<T> {
    @Override
    public T build(Class<T> clazz) {
        T object = null;  // 声明一个对象
        try {
            // 调用默认的构造函数创建一个对象，因为没法动态使用new生成对象
            object = (T)clazz.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
        return object;
    }
}

// 类似于c++的模版特例化，模版继承父类接口的时候为某个类型单独生成了一套代码
//public class AudiFactory implements Factory<Audi> {
//    @Override
//    public Audi build(Class<Audi> clazz) {
//        return new Audi();
//    }
//}

package com.example.demo;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

// 基于JDK实现动态代理
// 代理类：实现InvocationHandler接口
public class NewSell implements InvocationHandler {
    private Object object; // 被代理对象

    // 构造函数
    NewSell(Object object) {
        this.object = object;
    }

    // 处理实际方法调用：重写invoke方法，函数调用的本质还是内部调用invoke实现的
    // 扩展功能：在调用之前，加上日志
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("Long start NewSell");
        // 真正的函数调用，object.method的实现；这个方法是有返回值的，返回值可以为空
        Object res = method.invoke(object, args);
        System.out.println("Long finish NewSell");
        return res;
    }
}

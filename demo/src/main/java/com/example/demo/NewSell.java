package com.example.demo;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

// 基于JDK实现动态代理
public class NewSell implements InvocationHandler {
    private Object object; // 被代理对象

    NewSell(Object object) {
        this.object = object;
    }

    // 处理实际方法调用
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("Long start");
        Object res = method.invoke(object, args);
        System.out.println("Long finish");
        return res;
    }
}

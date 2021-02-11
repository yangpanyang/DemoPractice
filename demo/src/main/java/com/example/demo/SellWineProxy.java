package com.example.demo;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

// 基于JDK实现动态代理
// 代理类：实现InvocationHandler接口，和NewSell一模一样。。。
public class SellWineProxy implements InvocationHandler {
    private SellWine sellWine; // 被代理对象

    // 构造函数
    SellWineProxy(SellWine sellWine) {
        this.sellWine = sellWine;
    }

    // 重写invoke方法，函数调用的本质还是内部调用invoke实现的
    // 扩展功能：在调用之前，加上日志
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("Long start SellWineProxy");
        // 真正的函数调用，object.method的实现；这个方法是有返回值的，返回值可以为空
        Object res = method.invoke(sellWine, args);
        System.out.println("Long finish SellWineProxy");
        return res;
    }
}

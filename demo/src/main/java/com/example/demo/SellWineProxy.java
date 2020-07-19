package com.example.demo;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

public class SellWineProxy implements InvocationHandler {
    private SellWine sellWine;

    SellWineProxy(SellWine sellWine) {
        this.sellWine = sellWine;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("Long start");
        Object res = method.invoke(sellWine, args);
        System.out.println("Long finish");
        return res;
    }
}

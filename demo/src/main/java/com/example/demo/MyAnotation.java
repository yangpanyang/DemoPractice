package com.example.demo;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(value = {ElementType.TYPE, ElementType.METHOD, ElementType.FIELD})  // 元注解，可以用在哪些场合——可以用在type、method、field上
@Retention(value = RetentionPolicy.RUNTIME)  // 元注解，什么时候生效——运行时
public @interface MyAnotation {
    // 这个注解的两个属性
    String name();

    String value();
}

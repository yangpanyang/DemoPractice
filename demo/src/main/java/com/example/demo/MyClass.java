package com.example.demo;

@MyAnotation(name = "MyClass", value = "Class")  // ElementType.TYPE表示可以作用在这个类上
public class MyClass {
    @MyAnotation(name = "属性", value = "姓名")  // ElementType.FIELD表示可以作用在属性上，用注解说明属性的信息
    private String name;

    @MyAnotation(name = "属性", value = "年龄")
    private Integer age;

    MyClass(String name, Integer age) {
        this.name = name;
        this.age = age;
    }

    @MyAnotation(name = "方法", value = "显示")  // ElementType.METHOD表示可以作用在方法上
    public void display() {
        System.out.println(name + ": " + age);
    }

    //@MyAnotation(name = "方法", value = "getName")
    public String getName() {
        return name;
    }

    public Integer getAge() {
        return age;
    }
}

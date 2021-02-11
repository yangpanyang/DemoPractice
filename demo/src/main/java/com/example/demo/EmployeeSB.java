package com.example.demo;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component  // 注解1，加载的时候要扫描这个类
@ConfigurationProperties(prefix="employee")  // 注解2，配置对应的属性，默认路径在 resources/application.properties
public class EmployeeSB {
    String name;
    String gender;

    EmployeeSB() { }

    public void setName(String name) {
        this.name = name;
    }

    public String getName () {
        return name;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getGender () {
        return gender;
    }

    @Override
    public String toString() {
        return name + ": " + gender;
    }
}

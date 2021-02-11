package com.example.demo;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix="employee")
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

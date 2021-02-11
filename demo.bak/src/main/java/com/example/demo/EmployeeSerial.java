package com.example.demo;

import java.io.FileInputStream;
import java.io.Serializable;

public class EmployeeSerial implements Serializable {
    private static final long serialVersionUID = -76944442204051631L;

    private String id;
    private String name;
    private Integer age;
    private Integer salary;

    EmployeeSerial(String id, String name, Integer age, Integer salary) {
        this.id = id;
        this.name = name;
        this.age = age;
        this.salary = salary;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName () {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    public Integer getSalary () {
        return salary;
    }

    public void setSalary(Integer salary) {
        this.salary = salary;
    }

    @Override
    public String toString() {
        return id + "/" + name + ": " + age + ", " + salary;
    }
}

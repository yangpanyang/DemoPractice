package com.example.demo;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

public class Manager implements Approver {
    Approver next;
    ApproverObject nextObj;

    Manager() {
        nextObj =  buildObjectApprover();
    }

    @Override
    public void setNext(Approver next) {
        this.next = next;
    }

    @Override
    public boolean approve(String request) throws Exception {
        if (request.startsWith("M:")) {
            return true;
        } else {
            if (nextObj != null) {
                return nextObj.approve(request);
            } else {
                throw new Exception("No next to handle " + request);
            }
        }
    }

    // 依赖注入，代替setNext方法
    private ApproverObject buildObjectApprover() {
        ApproverObject approver = new ApproverObject();  // 创建对象
        // 注入属性
        Map<String, Class> map = new HashMap<>();  // 手动维护一张表，代替注解@Component的功能
        map.put("com.example.demo.Approver", com.example.demo.Director.class);  // (包的名字, 对应的实现)
        Field[] fields = approver.getClass().getDeclaredFields();  // 扫描所有属性
        for (Field field : fields) {
            try {
                String typeName = field.getType().getName();
                if (map.containsKey(typeName)) {
                    Class clazz = map.get(typeName);
                    field.setAccessible(true);
                    field.set(approver, clazz.getDeclaredConstructor().newInstance());  // 使用找到的类创建一个实例对象，依赖注入的核心
                }
            } catch (Exception e) {
                System.out.println("BuildObjectApprover exception: " + e);
            }
        }
        return approver;
    }
}

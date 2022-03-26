// -----JS CODE-----
//@ui{"widget":"separator"}
//@input SceneObject targetObject
global.ballTest = script;
script.vel = vec3.zero();
script.isThrown = false;

script.setVelocity = function (vel) {
    script.vel = vel;
    script.isThrown = true;
}

script.createEvent("UpdateEvent").bind(function () {
    if (getDeltaTime() <= 0) return;
    if (script.isThrown === false) return;
    
    var targetPosition = script.targetObject.getTransform().getWorldPosition();
    script.vel = script.vel.add(vec3.down().uniformScale(100*getDeltaTime()));
    var pos = script.getTransform().getWorldPosition();
    pos = pos.add(script.vel.uniformScale(getDeltaTime()));
//    print(pos)
    script.getTransform().setWorldPosition(pos);  
    if (pos['y'] < -50){
        script.isThrown = false;
    }
//    print(targetPosition)
    if ( Math.abs(targetPosition['x']-pos['x']) <10 && Math.abs(targetPosition['y']-pos['y']) <10 && Math.abs(targetPosition['z']-pos['z']) <10  ){
        print('yeah')
        print(targetPosition['x']-pos['x'])
        print(targetPosition['y']-pos['y'])        
        print(targetPosition['z']-pos['z'])
        print(pos)
        script.isThrown = false;
    }
   
});


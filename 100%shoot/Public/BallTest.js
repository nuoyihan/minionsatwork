// -----JS CODE-----
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
    script.vel = script.vel.add(vec3.down().uniformScale(300*getDeltaTime()));
    var pos = script.getTransform().getWorldPosition();
    pos = pos.add(script.vel.uniformScale(getDeltaTime()));
    script.getTransform().setWorldPosition(pos);    
});


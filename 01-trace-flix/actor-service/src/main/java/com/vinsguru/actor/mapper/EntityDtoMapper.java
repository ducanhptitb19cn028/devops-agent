package com.vinsguru.actor.mapper;

import com.vinsguru.actor.dto.ActorDto;
import com.vinsguru.actor.entity.Actor;

public class EntityDtoMapper {

    public static ActorDto toDto(Actor actor){
        return new ActorDto(
                actor.getId(),
                actor.getName()
        );
    }

}

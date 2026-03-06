package com.vinsguru.movie.dto;

import java.util.List;

public record MovieDto(Integer id,
                       String title,
                       Integer releaseYear,
                       List<ActorDto> actors,
                       List<ReviewDto> reviews) {
}

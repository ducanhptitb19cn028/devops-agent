package com.vinsguru.movie.mapper;


import com.vinsguru.movie.dto.ActorDto;
import com.vinsguru.movie.dto.MovieDto;
import com.vinsguru.movie.dto.ReviewDto;
import com.vinsguru.movie.entity.Movie;

import java.util.List;

public class EntityDtoMapper {

    public static MovieDto toDto(Movie movie, List<ActorDto> actors, List<ReviewDto> reviews){
        return new MovieDto(
                movie.getId(),
                movie.getTitle(),
                movie.getReleaseYear(),
                actors,
                reviews
        );
    }

}

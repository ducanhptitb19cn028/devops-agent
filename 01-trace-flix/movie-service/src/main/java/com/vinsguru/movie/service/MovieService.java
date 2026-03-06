package com.vinsguru.movie.service;

import com.vinsguru.movie.client.ActorClient;
import com.vinsguru.movie.client.ReviewClient;
import com.vinsguru.movie.dto.MovieDto;
import com.vinsguru.movie.entity.Movie;
import com.vinsguru.movie.mapper.EntityDtoMapper;
import com.vinsguru.movie.repository.MovieRepository;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class MovieService {

    private final MovieRepository repository;
    private final ActorClient actorClient;
    private final ReviewClient reviewClient;

    public MovieService(MovieRepository repository, ActorClient actorClient, ReviewClient reviewClient) {
        this.repository = repository;
        this.actorClient = actorClient;
        this.reviewClient = reviewClient;
    }

    public Optional<MovieDto> getMovie(Integer movieId) {
        return this.repository.findById(movieId)
                              .map(this::buildDto);
    }

    private MovieDto buildDto(Movie movie) {
        var reviews = this.reviewClient.getReviews(movie.getId());
        var actors = movie.getActorIds()
                          .stream()
                          .map(this.actorClient::getActor) // intentional sequential calls
                         //.gather(Gatherers.mapConcurrent(5, this.actorClient::getActor))
                          .toList();
        return EntityDtoMapper.toDto(movie, actors, reviews);
    }

}

"""Profiling script for the Tibetan Text Metrics tool."""

import cProfile
import pstats
from memory_profiler import profile
from tibetan_text_metrics.main import main
from tibetan_text_metrics.metrics import USE_CYTHON


@profile
def profile_main():
    """Run main function with memory profiling."""
    print(f"\nUsing {'Cython' if USE_CYTHON else 'Pure Python'} LCS implementation\n")
    main()


if __name__ == "__main__":
    # Run with cProfile and save stats
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the main function
    profile_main()
    
    profiler.disable()
    profiler.dump_stats('profile.stats')
    
    # Print CPU profiling results
    stats = pstats.Stats('profile.stats')
    print("\nTop 50 functions by cumulative time:")
    stats.sort_stats('cumulative').print_stats(50)
    
    print("\nTop 50 functions by total time:")
    stats.sort_stats('time').print_stats(50)
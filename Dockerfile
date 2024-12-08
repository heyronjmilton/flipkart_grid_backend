# Use the official Ubuntu base image
FROM ubuntu:latest



# Update the package list and install basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    vim \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables (optional)
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Set the working directory (optional)
WORKDIR /app



# Default command (can be overridden in `docker run`)
CMD ["/bin/bash"]


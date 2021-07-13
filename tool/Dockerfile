#FROM cuda9
#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update && apt install -y vim openssh-server telnet net-tools git zsh silversearcher-ag curl locales tmux
RUN mkdir /var/run/sshd
RUN echo 'root:$PASSWD' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && curl -sLO https://github.com/sharkdp/fd/releases/download/v7.3.0/fd-musl_7.3.0_amd64.deb && dpkg -i fd-musl_7.3.0_amd64.deb && locale-gen en_US.UTF-8

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile  && echo "export HOME=/home/sshadvr" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

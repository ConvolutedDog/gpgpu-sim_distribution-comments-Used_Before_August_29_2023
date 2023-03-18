// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*
有两种传统方法可以检测传统CPU体系结构中指令之间的相关性：记分牌和保留站。保留站用于消除名称依赖性，
并引入对关联逻辑的需要，而关联逻辑在面积和能量方面是昂贵的。记分牌可以设计为支持按序执行或乱序执行。
支持乱序执行的记分牌（如CDC 6600中使用的记分牌）也相当复杂。另一方面，单线程按序的CPU中的记分牌非
常简单：在记分牌中用单个位来表示每一个寄存器，每当发出将写入到该寄存器的指令时，记分牌中对应的单个
位被设定。任何想要读取或写入在记分牌中设置了相应位的寄存器的指令都会stall，直到写入寄存器的指令清
除了该位。这可以防止写后读和写后写危险。如果寄存器文件的read被限制为按顺序发生，这是按CPU设计中的
典型情况，则与按顺序指令发射相结合时，这种简单的记分牌可以防止读后写故障。考虑到这是最简单的设计，
因此将消耗最少的面积和能源，GPU实现了按顺序记分牌。在支持多个warp时，使用按顺序记分牌存在一些挑战。
*/

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include "assert.h"

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

class Scoreboard {
 public:
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);

  void reserveRegisters(const warp_inst_t *inst);
  void releaseRegisters(const warp_inst_t *inst);
  void releaseRegister(unsigned wid, unsigned regnum);

  bool checkCollision(unsigned wid, const inst_t *inst) const;
  bool pendingWrites(unsigned wid) const;
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);

 private:
  void reserveRegister(unsigned wid, unsigned regnum);
  int get_sid() const { return m_sid; }

  unsigned m_sid;

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  //跟踪对寄存器的写入挂起。
  //按warp id索引，reg_id=>挂起的写入计数。
  std::vector<std::set<unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)
  std::vector<std::set<unsigned> > longopregs;

  class gpgpu_t *m_gpu;
};

#endif /* SCOREBOARD_H_ */

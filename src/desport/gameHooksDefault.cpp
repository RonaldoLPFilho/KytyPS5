// SPDX-License-Identifier: GPL-2.0-only
// desport: used when the engine is built without the desport game layer (DESPORT_GAME_DIR unset).
#include "loader/desport.h"

namespace Desport {

void RegisterGameHooks(Loader::SymbolDatabase* /*symbols*/, const std::string& /*title_id*/) {}

} // namespace Desport

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnIconFactory.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Factory for creating icons for nodes. Icons are associated with a node's
**  type (so all nodes of the same type have the same icon). Unknown types
**  get a generic icon. The colour of the top-left corner of the icon is
**  assumed to be transparent.
**
*****************************************************************************/

#pragma once

#include "wx/wx.h"
#include "wx/utils.h"

// drwnIconFactory -----------------------------------------------------------

class drwnIconFactory
{
 protected:
    wxBitmap *_defaultIcon;
    map<string, wxBitmap *> _registry;

 public:    
    ~drwnIconFactory();
    static drwnIconFactory& get();

    void registerIcon(const char *name, const wxBitmap *bmp);
    void unregisterIcon(const char *name);

    const wxBitmap *getIcon(const char *name) const;
    
 protected:
    drwnIconFactory(); // singleton class (requires gui initialization)

    // Creates the transparency mask for the icon by examining the colour
    // of the top-left corner. Returns the same pointer passed in.
    static wxBitmap *setIconMask(wxBitmap *bmp);
};

#define gIconFactory drwnIconFactory::get()
